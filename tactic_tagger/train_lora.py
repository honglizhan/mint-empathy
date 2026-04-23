import os
import re
import torch
import wandb
import numpy as np
from absl import app, flags
from datasets import load_dataset
from termcolor import cprint, colored
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from functools import partial

FLAGS = flags.FLAGS
flags.DEFINE_string("model_id", "meta-llama/Llama-3.1-8B-Instruct", "")
flags.DEFINE_string("train_data_path", '../data/tagger_annotations/train/advice.jsonl', "")
flags.DEFINE_string("val_data_path", '../data/tagger_annotations/val/advice.jsonl', "")
flags.DEFINE_string("finetuned_model_path", "./trained-tactic-tagger-models-lora_adapters/Llama-3.1-8B-Instruct-tagger-advice", "")
flags.DEFINE_string("ckpt_output_path", "./ckpt/Llama-3.1-8B-Instruct-tagger-advice", "")
flags.DEFINE_float("training_epochs", 3.0, "")
flags.DEFINE_integer("batch_size", 128, "")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16
print (f"use_bf16: {use_bf16}, use_fp16: {use_fp16}")


def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits to save memory during evaluation.
    Convert logits to predicted token IDs on GPU before accumulation.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    # Predictions are already argmax'd by preprocess_logits_for_metrics
    predictions = predictions.astype(np.int64)

    # Replace -100 in predictions and labels with pad token
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    decoded_labels = []
    for i in range(len(labels)):
        valid_tokens = labels[i][labels[i] != -100]
        if len(valid_tokens) > 0:
            decoded_labels.append(tokenizer.decode(valid_tokens, skip_special_tokens=True))
        else:
            decoded_labels.append("")

    # Extract scores from <score> tags
    pred_labels = []
    true_labels = []
    invalid_labels = []

    for pred, label in zip(decoded_preds, decoded_labels):
        label_match = re.search(r'<score>(\d+)</score>', label)
        if not label_match:
            invalid_labels.append(label)
            continue

        pred_match = re.search(r'<score>(\d+)</score>', pred)
        pred_labels.append(int(pred_match.group(1)) if pred_match else 0)
        true_labels.append(int(label_match.group(1)))

    # Show first 10 predicted labels and count valid predictions
    total_samples = len(decoded_preds)
    valid_samples = len(pred_labels)

    cprint(f"\n=== Evaluation Metrics Debug Info ===", "cyan")
    cprint(f"Total samples in batch: {total_samples}", "yellow")
    cprint(f"Valid samples (with score tags): {valid_samples}", "green")
    cprint(f"Invalid samples: {total_samples - valid_samples}", "red")

    cprint(f"\nInvalid labels:", "cyan")
    for i in range(min(10, len(invalid_labels))):
        cprint(f"  Sample {i}: pred={invalid_labels[i]}, true={true_labels[i]}", "white")

    cprint(f"=====================================\n", "cyan")

    if not pred_labels or not true_labels:
        return {"accuracy": 0.0, "f1": 0.0}

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro', pos_label=1)

    return {"accuracy": accuracy, "f1": f1}


def prepare_data(my_tokenizer, train_data_path, val_data_path):
    ## ---- preparing data ---
    # def formatting_prompts_func(examples):
    #     convos = examples["conversations"]
    #     texts = [my_tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    #     return { "text" : texts, }
    # pass

    data_files = {
        'train': train_data_path,
        'validation': val_data_path,
    }

    dataset = load_dataset('json', data_files=data_files)
    # dataset = dataset.rename_column("messages", "conversations")
    # dataset = dataset.map(formatting_prompts_func, batched = True,)
    return dataset


def main(argv):

    if os.path.exists(FLAGS.finetuned_model_path):
        raise RuntimeError(colored(f"The directory {FLAGS.finetuned_model_path} already exists. The model is already trained, so no further action is needed.", "red"))

    wandb.init(
        project = "empathy-tactics-multi-turn",
        name    = FLAGS.finetuned_model_path.split('/')[-1],
    )

    ## ---- load un-finetuned model ---
    model = AutoModelForCausalLM.from_pretrained(
        FLAGS.model_id,
        device_map = "auto",
        dtype      = torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "llama-3" in FLAGS.model_id.lower():
        tokenizer.chat_template = """{% for message in messages %}{{'<|begin_of_text|>' if loop.first else ''}}{% if message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|eot_id|>' }}{% endif %}{% endfor %}"""
    else:
        raise RuntimeError(f"Chat template not supported for {FLAGS.model_id}. Please add chat template.")

    peft_config = LoraConfig(
        r               = 64,
        lora_alpha      = 32,
        lora_dropout    = 0.05,
        bias            = "none",
        target_modules  = "all-linear",
        task_type       = "CAUSAL_LM",
    )

    args = SFTConfig(
        num_train_epochs            = FLAGS.training_epochs,
        per_device_train_batch_size = FLAGS.batch_size,
        per_device_eval_batch_size  = 4,
        gradient_accumulation_steps = 4,
        warmup_steps                = 5,
        max_steps                   = -1,
        learning_rate               = 5e-5,
        max_grad_norm               = 1.0,
        warmup_ratio                = 0.1,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        logging_steps               = 1,
        optim                       = "adamw_torch_fused",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        output_dir                  = FLAGS.ckpt_output_path,
        logging_dir                 = f"{FLAGS.ckpt_output_path}/logs",
        logging_strategy            = "steps",
        eval_strategy               = "steps",
        save_strategy               = "steps",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        eval_steps                  = 10,
        save_steps                  = 10,
        save_total_limit            = 2,
        report_to                   = "wandb",
        assistant_only_loss         = True,
        gradient_checkpointing      = True,
        # dataset_text_field          = "text",
        max_length                  = 1024*4,
    )

    dataset_prepared = prepare_data(tokenizer, FLAGS.train_data_path, FLAGS.val_data_path)

    cprint (dataset_prepared["validation"][5]["messages"], "yellow")

    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)

    ## ---- finetune the model ---
    trainer = SFTTrainer(
        model               = model,
        processing_class    = tokenizer,
        peft_config         = peft_config,
        # data_collator       = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        train_dataset       = dataset_prepared["train"],
        eval_dataset        = dataset_prepared["validation"],  # Add validation dataset
        compute_metrics     = compute_metrics_with_tokenizer,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        args                = args,
    )

    # We verify masking is actually done:
    cprint ("\n--- Example input and labels ---", "yellow")
    data_collator = trainer.data_collator
    data_iterator = iter(trainer.get_train_dataloader())
    batch = next(data_iterator)

    # Decode and print the first example in the batch
    cprint(tokenizer.decode(batch["input_ids"][0]), "cyan")
    cprint("------", "cyan")

    # Print the labels for the first example
    labels = batch["labels"][0]
    cprint(labels, "green")


    cprint ("\n--- Trainable Parameters ---", "yellow")
    trainer.model.print_trainable_parameters()
    cprint ("------", "yellow")

    trainer.train()
    eval_results = trainer.evaluate()
    print (eval_results)

    ## ---- save model ---
    if not os.path.exists(FLAGS.finetuned_model_path):
        os.makedirs(FLAGS.finetuned_model_path)

    trainer.save_model(FLAGS.finetuned_model_path)
    cprint(f"Model and tokenizer saved to {FLAGS.finetuned_model_path}", "yellow")


if __name__ == "__main__":
    app.run(main)
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links = []
        self.nodes = []
        self.current_link_index = None
        self.current_link_text = []

    def handle_starttag(self, tag, attrs) -> None:
        attr_map = {key: value or '' for key, value in attrs}
        self.nodes.append((tag, attr_map))
        if tag == 'a':
            attr_map['_text'] = ''
            self.links.append(attr_map)
            self.current_link_index = len(self.links) - 1
            self.current_link_text = []

    def handle_data(self, data) -> None:
        if self.current_link_index is not None:
            self.current_link_text.append(data)

    def handle_endtag(self, tag) -> None:
        if tag != 'a' or self.current_link_index is None:
            return
        text = ' '.join(' '.join(self.current_link_text).split())
        self.links[self.current_link_index]['_text'] = text
        self.current_link_index = None
        self.current_link_text = []


class ProjectPageStructureTest(unittest.TestCase):
    def read_html(self) -> str:
        return Path('project-page/index.html').read_text(encoding='utf-8')

    def parse_html(self) -> LinkParser:
        parser = LinkParser()
        parser.feed(self.read_html())
        return parser

    def test_page_uses_live_publication_links(self) -> None:
        html = self.read_html()
        self.assertIn('https://arxiv.org/abs/2604.11742', html)
        self.assertIn('https://arxiv.org/pdf/2604.11742', html)
        self.assertIn('citation_arxiv_id', html)
        self.assertNotIn('Paper and data will be available upon publication', html)
        self.assertNotIn('coming soon', html.lower())
        self.assertNotIn('topbar__pill--disabled', html)
        self.assertNotIn('hero__btn--disabled', html)

    def test_page_links_to_released_artifacts(self) -> None:
        html = self.read_html()
        self.assertIn('https://huggingface.co/hongli-zhan/MINT-empathy-Qwen3-4B', html)
        self.assertIn('https://huggingface.co/hongli-zhan/MINT-empathy-Qwen3-1.7B', html)
        self.assertIn('https://huggingface.co/hongli-zhan/empathy-tactic-taggers-llama3.1-8b', html)
        self.assertIn('Tag your conversation', html)
        self.assertIn('Open in Colab', html)

    def test_page_highlights_problem_method_and_results(self) -> None:
        html = self.read_html()
        self.assertIn('Problem', html)
        self.assertIn('Metric', html)
        self.assertIn('Method', html)
        self.assertIn('Result', html)
        self.assertIn('25.3%', html)
        self.assertIn('26.3%', html)
        self.assertIn('0.50 to 0.56', html)
        self.assertIn('0.27', html)


    def test_primary_ctas_show_all_model_artifacts(self) -> None:
        html = self.read_html()
        self.assertIn('4B Model', html)
        self.assertIn('1.7B Model', html)
        self.assertIn('Tactic Taggers', html)
        self.assertIn('Colab', html)

    def test_hero_does_not_duplicate_paper_with_pdf_cta(self) -> None:
        parser = self.parse_html()
        hero_labels = [
            attrs.get('_text', '')
            for attrs in parser.links
            if 'hero__btn' in attrs.get('class', '')
        ]
        self.assertIn('Paper', hero_labels)
        self.assertNotIn('PDF', hero_labels)
        self.assertIn('citation_pdf_url', self.read_html())

    def test_chart_uses_tactic_stickiness_axis(self) -> None:
        html = self.read_html()
        self.assertIn('Tactic Stickiness', html)
        self.assertIn('stickiness, empathy, name, family, model_size', html)
        self.assertIn("[0.42, 4.67, 'MINT', 'ours', '4B']", html)
        self.assertIn("[0.27, 2.90, 'Human (Gold)', 'human', 'both']", html)
        self.assertNotIn('Tactic Diversity (more diverse', html)

    def test_chart_matches_paper_axis_direction(self) -> None:
        html = self.read_html()
        self.assertIn('reverse: true', html)
        self.assertIn('Tactic Stickiness \\u2192 more diverse', html)
        self.assertIn('MINT moves models toward the upper right', html)
        self.assertNotIn('MINT moves models toward the upper left', html)

    def test_chart_highlights_mint_region_like_paper(self) -> None:
        html = self.read_html()
        self.assertIn('idealRegionPlugin', html)
        self.assertIn('MINT target region', html)
        self.assertIn('xMin: 0.37', html)
        self.assertIn('xMax: 0.55', html)
        self.assertIn('yMin: 4.42', html)
        self.assertIn('yMax: 4.80', html)
        self.assertIn("logo.src = 'static/mint_logo.png'", html)
        self.assertIn('This green region marks the high empathy, low stickiness zone containing both MINT models.', html)

    def test_chart_logo_uses_mobile_safe_anchor(self) -> None:
        html = self.read_html()
        self.assertIn('function getMintLogoPosition(chart, idealRegion, size)', html)
        self.assertIn('if (isSmallScreen) {', html)
        self.assertIn('var mobileX = right + size / 2 + 8', html)
        self.assertIn('var mobileY = top + size / 2 + 8', html)
        self.assertIn('var logoPosition = getMintLogoPosition(chart, idealRegion, size)', html)
        self.assertIn('var x = logoPosition.x', html)
        self.assertIn('var y = logoPosition.y', html)
        self.assertNotIn('var x = right - size / 2 - 10', html)
        self.assertNotIn('var y = top + size / 2 + 10', html)
        self.assertNotIn('var x = xScale.getPixelForValue(0.535)', html)
        self.assertNotIn('var y = yScale.getPixelForValue(4.73)', html)

    def test_human_gold_tooltip_omits_model_size(self) -> None:
        html = self.read_html()
        self.assertIn("if (meta.size !== 'both')", html)
        self.assertIn("methodLabel += ' (' + meta.size + ')'", html)
        self.assertIn("return methodLabel + ': Empathy='", html)
        self.assertNotIn("meta.name + ' (' + meta.size + '): Empathy='", html)

    def test_pareto_chart_has_mobile_layout(self) -> None:
        html = self.read_html()
        self.assertIn('chart-mobile-legend', html)
        self.assertIn("window.matchMedia('(max-width: 640px)')", html)
        self.assertIn('aspectRatio: isSmallScreen ? 0.9 : 1.5', html)
        self.assertIn('display: !isSmallScreen', html)
        self.assertIn("title: { display: !isSmallScreen, text: 'Aggregate Empathy", html)
        self.assertIn('pointRadius: getPointRadius(fam)', html)

    def test_page_has_try_it_yourself_tools(self) -> None:
        html = self.read_html()
        self.assertIn('Try it yourself', html)
        self.assertIn('Tag your own conversation', html)
        self.assertIn('Plot your outputs on the Pareto front', html)
        self.assertIn('analysis/tagger_colab_starter.ipynb', html)

    def test_long_conversation_examples_are_collapsed(self) -> None:
        html = self.read_html()
        self.assertIn('<details class="conversation-details">', html)
        self.assertIn('View conversation examples', html)

    def test_website_source_contains_no_cjk_text(self) -> None:
        cjk_pattern = re.compile(
            r'[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff'
            r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
            r'\uff00-\uffef\uac00-\ud7af]'
        )
        text_suffixes = {'.html', '.css', '.js', '.svg', '.md', '.txt', '.json'}
        findings = []
        for path in sorted(Path('project-page').rglob('*')):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            text = path.read_text(encoding='utf-8')
            for line_number, line in enumerate(text.splitlines(), 1):
                if cjk_pattern.search(line):
                    findings.append(f'{path}:{line_number}: {line.strip()}')

        self.assertEqual([], findings)

    def test_no_inline_styles_or_hrefless_cta_links(self) -> None:
        parser = self.parse_html()
        inline_style_nodes = [tag for tag, attrs in parser.nodes if 'style' in attrs]
        self.assertEqual([], inline_style_nodes)

        cta_links_without_href = [
            attrs.get('class', '')
            for attrs in parser.links
            if 'hero__btn' in attrs.get('class', '') and not attrs.get('href')
        ]
        self.assertEqual([], cta_links_without_href)


if __name__ == '__main__':
    unittest.main()

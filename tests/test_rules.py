import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest.mock import patch

from observatory.__main__ import main
from observatory.ingest import label_key
from observatory.rules import classify_rules, compare, normalize, predict
from observatory.store import digest, read_json, read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(ROOT / 'config/observatory.json')
        self.book = read_json(ROOT / 'config/paper_rules.json')
        write_json(self.root / 'config/observatory.json', self.config)
        write_json(self.root / 'config/paper_rules.json', self.book)
        self.today = date(2026, 9, 13)

    def paper(self, pid, title, abstract):
        return {'id': pid, 'title': title, 'abstract': abstract, 'content_hash': digest([title, abstract])}

    def test_general_topic_signals_and_boundaries(self):
        examples = [
            ('Benchmarking robot agents', 'We introduce a benchmark for measuring robotic agents and their tool use.', 'evaluation'),
            ('Quantization for vision-language models', 'Low-bit quantization compresses video models, reducing memory overhead during inference.', 'systems'),
            ('World generation from photos', 'We reconstruct static scenes as individual object meshes from a single reference image.', 'multimodal'),
            ('Action-conditioned world models', 'We introduce learned simulators of environment dynamics to predict future states and action outcomes.', 'world_models'),
            ('Whole-body robot control', 'Our robotic policy performs humanoid navigation and manipulation.', 'robotics'),
            ('Tool-using agent orchestration', 'A multi-agent system selects external tools and coordinates planning and execution.', 'agents'),
            ('On-policy self-distillation', 'A new training method improves mathematical reasoning through policy optimization.', 'reasoning'),
            ('A catalog of minerals', 'We describe sediment samples and local mineral deposits.', 'other'),
        ]
        for title, abstract, expected in examples:
            with self.subTest(title=title):
                self.assertEqual(predict(self.paper('unseen', title, abstract), self.book)['category'], expected)

    def test_repetition_unicode_and_missing_evidence(self):
        p = self.paper('first', 'Audio–encoder compression', 'Quantization reduces memory overhead.')
        normal = predict(p, self.book)
        repeated = predict({**p, 'id': 'different', 'abstract': p['abstract'] * 20}, self.book)
        self.assertEqual(normal['scores'], repeated['scores'])
        self.assertEqual(normalize('WORLD‑ACTION'), 'world action')
        unknown = predict(self.paper('unknown', 'Untitled', ''), self.book)
        self.assertEqual(unknown['category'], 'other')
        self.assertTrue(unknown['uncertain'])
        self.assertTrue(unknown['fallback'])

    def test_shadow_preserves_references_and_apply_fills_only_missing(self):
        p1 = self.paper('2609.00001', 'Robot control', 'A policy enables robotic manipulation.')
        p2 = self.paper('2609.00002', 'Fast quantization', 'Low-bit compression reduces memory.')
        write_jsonl(self.root / 'data/papers.jsonl', [p1, p2])
        reference = {'key': label_key(p1, self.config), 'id': p1['id'], 'category': 'evaluation',
                     'source': 'assistant_bootstrap', 'uncertain': True}
        path = self.root / 'data/labels.jsonl'
        write_jsonl(path, [reference])
        original = path.read_bytes()
        with patch('requests.Session.request', side_effect=AssertionError('Rules cannot call the network')):
            predictions, stats = classify_rules(self.root, self.config, self.today)
            self.assertEqual(path.read_bytes(), original)
            metrics = compare(self.root, self.config, predictions)
            self.assertEqual(metrics['compared'], 1)
            self.assertEqual(metrics['matched'], 0)
            self.assertEqual(metrics['excluded_without_reference'], 1)
            _, stats = classify_rules(self.root, self.config, self.today, apply=True)
        labels = {l['id']: l for l in read_jsonl(path)}
        self.assertEqual(labels[p1['id']], reference)
        self.assertEqual(labels[p2['id']]['source'], 'keyword_rules')
        self.assertEqual(stats['labels_applied'], 1)
        self.assertEqual(stats['classifier_requests'], 0)
        _, stats = classify_rules(self.root, self.config, self.today, apply=True)
        self.assertEqual(stats['rules_recomputed'], 0)
        self.assertEqual(stats['labels_applied'], 0)

    def test_editing_rules_invalidates_cache_and_updates_rule_labels(self):
        paper = self.paper('2609.00001', 'A specialtopic method', 'An otherwise unknown subject.')
        write_jsonl(self.root / 'data/papers.jsonl', [paper])
        classify_rules(self.root, self.config, self.today, apply=True)
        changed = copy.deepcopy(self.book)
        changed['rules']['agents'].append(['new topic', 10, r'\bspecialtopic\b'])
        write_json(self.root / 'config/paper_rules.json', changed)
        _, stats = classify_rules(self.root, self.config, self.today, apply=True)
        self.assertEqual(stats['rules_recomputed'], 1)
        self.assertEqual(stats['labels_applied'], 1)
        self.assertEqual(read_jsonl(self.root / 'data/labels.jsonl')[0]['category'], 'agents')

    def test_default_cli_never_reads_token_or_constructs_hf_client(self):
        p = self.paper('2609.00001', 'Policy optimization', 'Reinforcement learning for mathematical reasoning.')
        write_jsonl(self.root / 'data/papers.jsonl', [p])
        for command in ['classify', 'compare-labels']:
            with self.subTest(command=command), patch('sys.argv', ['observatory', command, '--root', str(self.root)]), \
                    patch('observatory.__main__.load_token', side_effect=AssertionError('No token needed')), \
                    patch('observatory.__main__.HFClient', side_effect=AssertionError('No HF client needed')), redirect_stdout(io.StringIO()):
                main()


if __name__ == '__main__':
    unittest.main()

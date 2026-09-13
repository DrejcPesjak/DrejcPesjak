import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from observatory.ingest import Deferred, HFClient, canonical_id, classify, collect_models, collect_papers, label_key
from observatory.render import render, reconcile_records, scaffold
from observatory.store import completed_dates, digest, read_json, read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, data=None, status=200, headers=None, links=None):
        self.data, self.status_code = data, status
        self.headers, self.links = headers or {}, links or {}
        self.ok, self.is_redirect = status < 400, False

    def json(self):
        return self.data


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls, self.budget = iter(responses), 0, 30
        self.usage = {'input_tokens': 0, 'output_tokens': 0}

    def request(self, *args, **kwargs):
        self.calls += 1
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


class ObservatoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(ROOT / 'config/observatory.json')
        self.today = date(2026, 9, 13)

    def paper(self, pid='2609.00001', title='Robot navigation'):
        return {'id': pid, 'title': title, 'abstract': 'Learning robot navigation.',
                'url': 'https://huggingface.co/papers/' + pid, 'content_hash': digest(title)}

    def test_completed_dates_and_versions(self):
        self.assertEqual(completed_dates(self.today, 7), [f'2026-09-{d:02}' for d in range(6, 13)])
        self.assertEqual(canonical_id('2609.00001v2'), '2609.00001')
        with self.assertRaises(ValueError):
            canonical_id('<script>')

    def test_keyed_svg_preserves_custom_scaffold(self):
        svg = scaffold('models').replace('<defs>', '<!-- custom artwork stays -->\n<defs>')
        svg, result = reconcile_records(svg, {'abcd': '<circle id="old"/>'})
        svg2, result = reconcile_records(svg, {'abcd': '<circle id="old"/>', 'ef01': '<circle id="new"/>'})
        self.assertEqual(result, {'added': 1, 'updated': 0, 'removed': 0})
        self.assertIn('<!-- custom artwork stays -->', svg2)
        self.assertIn('<!--record:abcd--><circle id="old"/><!--/record:abcd-->', svg2)
        same, result = reconcile_records(svg2, {'abcd': '<circle id="old"/>', 'ef01': '<circle id="new"/>'})
        self.assertEqual(same, svg2)

    def test_pagination_dedupe_empty_and_cached_rerun(self):
        self.config['window_days'] = 1
        row = {'paper': {'id': '2609.00001v2', 'title': 'Paper', 'summary': 'Abstract'}}
        client = FakeClient([Response([row, row], links={'next': {'url': 'https://huggingface.co/api/daily_papers?date=2026-09-12&p=1'}}), Response([])])
        self.assertEqual(collect_papers(self.root, self.config, client, self.today), [])
        day = read_json(self.root / 'data/days.json')['2026-09-12']
        self.assertEqual(day['ids'], ['2609.00001'])
        self.assertTrue(day['complete'])
        client = FakeClient([])
        collect_papers(self.root, self.config, client, self.today)
        self.assertEqual(client.calls, 0)

    def test_failed_and_truncated_fetch_are_not_zero(self):
        self.config['window_days'] = 1
        client = FakeClient([Deferred('rate limited')])
        self.assertTrue(collect_papers(self.root, self.config, client, self.today))
        self.assertFalse((self.root / 'data/days.json').exists())
        self.config['paper_pages_per_day'] = 1
        row = {'paper': {'id': '2609.00001', 'title': 'Paper', 'summary': ''}}
        client = FakeClient([Response([row], links={'next': {'url': 'https://huggingface.co/api/daily_papers?date=2026-09-12&p=1'}})])
        self.assertTrue(collect_papers(self.root, self.config, client, self.today))
        self.assertFalse((self.root / 'data/days.json').exists())

    def test_today_is_partial_and_refetched_when_completed(self):
        self.config['window_days'] = 1
        first = {'paper': {'id': '2609.00001', 'title': 'First', 'summary': ''}}
        second = {'paper': {'id': '2609.00002', 'title': 'Late arrival', 'summary': ''}}
        client = FakeClient([Response([]), Response([first])])
        self.assertEqual(collect_papers(self.root, self.config, client, self.today, include_today=True), [])
        current = read_json(self.root / 'data/days.json')['2026-09-13']
        self.assertFalse(current['complete'])
        self.assertEqual(current['ids'], ['2609.00001'])
        client = FakeClient([Response([first, second])])
        self.assertEqual(collect_papers(self.root, self.config, client, date(2026, 9, 14)), [])
        completed = read_json(self.root / 'data/days.json')['2026-09-13']
        self.assertTrue(completed['complete'])
        self.assertEqual(len(completed['ids']), 2)

    def test_model_scope_exclusions_and_snapshot_cache(self):
        rows = [
            {'id': 'Qwen/Good', 'downloads': 1000, 'safetensors': {'total': 8000000000}, 'pipeline_tag': 'text-generation'},
            {'id': 'another/Omni', 'downloads': 2000, 'safetensors': {'total': 9000000000}, 'pipeline_tag': 'any-to-any'},
            {'id': 'hf-internal-testing/Test', 'downloads': 100000, 'safetensors': {'total': 100}, 'pipeline_tag': 'text-generation'},
            {'id': 'lab/UnknownSize', 'downloads': 10000, 'pipeline_tag': 'text-generation'},
            {'id': 'lab/Quantized', 'downloads': 10000, 'safetensors': {'total': 1000}, 'tags': ['gguf'], 'pipeline_tag': 'text-generation'},
        ]
        client = FakeClient([Response(rows) for _ in range(6)])
        self.assertEqual(collect_models(self.root, self.config, client, self.today), [])
        snapshot = read_json(self.root / 'data/models.json')
        self.assertEqual({m['id'] for m in snapshot['models']}, {'Qwen/Good', 'another/Omni'})
        self.assertEqual(len(snapshot['excluded']), 3)
        self.assertEqual(next(m['family'] for m in snapshot['models'] if m['id'] == 'Qwen/Good'), 'Qwen / Alibaba')
        client = FakeClient([])
        self.assertEqual(collect_models(self.root, self.config, client, self.today), [])
        self.assertEqual(client.calls, 0)

    def test_thirty_day_window_and_busy_day_aggregation(self):
        self.config['window_days'] = 30
        papers = [self.paper(f'2609.{i:05}') for i in range(451)]
        write_jsonl(self.root / 'data/papers.jsonl', papers)
        write_json(self.root / 'data/days.json', {'2026-09-12': {'complete': True, 'ids': [p['id'] for p in papers]}})
        render(self.root, self.config, self.today)
        svg = (self.root / 'assets/papers.svg').read_text()
        self.assertIn('30 days in research', svg)
        self.assertIn('marks aggregate category counts', svg)
        self.assertIn('451 APPEARANCES', svg)
        self.assertLess(len(svg.encode()), 200000)

    def test_classifier_cache_and_only_invalid_items_repaired(self):
        papers = [self.paper(), self.paper('2609.00002', 'Tool agents')]
        write_jsonl(self.root / 'data/papers.jsonl', papers)
        def answer(rows):
            return Response({'choices': [{'message': {'content': json.dumps({'labels': rows})}}]})
        client = FakeClient([
            answer([{'id': papers[0]['id'], 'category': 'robotics', 'uncertain': False, 'reason': 'Control'}]),
            answer([{'id': papers[1]['id'], 'category': 'agents', 'uncertain': False, 'reason': 'Tools'}]),
        ])
        self.assertEqual(classify(self.root, self.config, client, 'test-token', self.today), [])
        self.assertEqual(client.calls, 2)
        client = FakeClient([])
        classify(self.root, self.config, client, 'test-token', self.today)
        self.assertEqual(client.calls, 0)
        self.assertEqual(len(read_jsonl(self.root / 'data/labels.jsonl')), 2)
        changed = {**papers[0], 'content_hash': 'new'}
        self.assertNotEqual(label_key(papers[0], self.config), label_key(changed, self.config))

    def test_renderer_is_offline_idempotent_and_rolls_window(self):
        paper = self.paper(title='Title with < & characters')
        write_jsonl(self.root / 'data/papers.jsonl', [paper])
        write_json(self.root / 'data/days.json', {'2026-09-06': {'complete': True, 'ids': [paper['id']]}, '2026-09-07': {'complete': True, 'ids': []}})
        with patch('requests.Session.request', side_effect=AssertionError('render must be offline')):
            render(self.root, self.config, self.today)
            original = (self.root / 'assets/papers.svg').read_bytes()
            result = render(self.root, self.config, self.today)
            self.assertFalse(result['papers']['changed'])
            self.assertEqual(original, (self.root / 'assets/papers.svg').read_bytes())
            svg = original.decode()
            self.assertIn('1 APPEARANCES', svg)
            self.assertIn('NO DATA', svg)
            self.assertIn('Title with &lt; &amp; characters', svg)
            render(self.root, self.config, date(2026, 9, 14))
            self.assertNotIn('day-2026-09-06', (self.root / 'assets/papers.svg').read_text())
            self.assertEqual(len(read_jsonl(self.root / 'data/papers.jsonl')), 1)

    @patch('observatory.ingest.time.sleep')
    def test_request_ceiling_and_long_retry_after(self, sleep):
        from unittest.mock import Mock
        session = Mock()
        session.headers = {}
        session.request.return_value = Response(status=429, headers={'Retry-After': '120'})
        client = HFClient(None, 2, 0, session)
        with self.assertRaises(Deferred):
            client.request('GET', 'https://huggingface.co/api/models')
        self.assertEqual(client.calls, 1)
        sleep.assert_not_called()
        session.request.return_value = Response(status=503)
        with self.assertRaises(Deferred):
            client.request('GET', 'https://huggingface.co/api/models')
        self.assertEqual(client.calls, 2)
        with self.assertRaises(ValueError):
            client.request('GET', 'https://example.com/')


if __name__ == '__main__':
    unittest.main()

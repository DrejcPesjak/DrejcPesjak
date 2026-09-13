import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from .ingest import HFClient, classify, collect_models, collect_papers
from .render import render
from .rules import classify_rules, compare
from .store import load_token, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description='Update a persistent HF observatory.')
    parser.add_argument('command', choices=['update', 'ingest', 'classify', 'compare-labels', 'render'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--today', type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument('--refresh', action='store_true', help='Explicitly refetch cached completed dates and model snapshot')
    parser.add_argument('--include-today', action='store_true', help='Also collect today as partial; charts still show completed days')
    parser.add_argument('--max-batches', type=int, help='Bound initial label pilot; successful labels are cached')
    parser.add_argument('--backend', choices=['rules', 'hf'], help='Classification backend; defaults to config (rules)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    config = read_json(args.root / 'config/observatory.json')
    if not 1 <= config['window_days'] <= 30:
        parser.error('window_days must be between 1 and 30')
    backend = args.backend or config.get('classification_backend', 'rules')
    if backend not in {'rules', 'hf'}:
        parser.error('classification_backend must be rules or hf')
    token = load_token() if args.command in {'update', 'ingest'} or (args.command == 'classify' and backend == 'hf') else None
    errors, stats = [], {}
    if args.command in {'update', 'ingest'}:
        client = HFClient(token, config['hf_request_budget'], config['request_spacing_seconds'])
        errors += collect_models(args.root, config, client, args.today, args.refresh)
        errors += collect_papers(args.root, config, client, args.today, args.refresh, args.include_today)
        stats['hub_requests'] = client.calls
    if args.command in {'update', 'classify'}:
        stats['classification_backend'] = backend
        if backend == 'rules':
            _, rule_stats = classify_rules(args.root, config, args.today, apply=True)
            stats.update(rule_stats)
        else:
            client = HFClient(token, config['classifier_request_budget'], config['request_spacing_seconds'])
            errors += classify(args.root, config, client, token, args.today, args.max_batches)
            stats['classifier_requests'] = client.calls
            stats.update(client.usage)
    if args.command == 'compare-labels':
        predictions, rule_stats = classify_rules(args.root, config, args.today)
        stats.update(rule_stats)
        stats['comparison'] = compare(args.root, config, predictions)
    if args.command in {'update', 'render'}:
        stats['svg_patches'] = render(args.root, config, args.today)
    if args.command not in {'render', 'compare-labels'} and (errors or stats.get('hub_requests') or stats.get('classifier_requests') or stats.get('rule_predictions')):
        write_json(args.root / 'data/status.json', {'date': args.today.isoformat(), 'errors': errors, **stats})
    print(json.dumps({'errors': errors, **stats}, indent=2))
    # Partial runs intentionally retain/publish valid data; visible status and job
    # warnings make deferrals observable without discarding successful batches.
    for error in errors:
        logging.warning('Deferred: %s', error)


if __name__ == '__main__':
    main()

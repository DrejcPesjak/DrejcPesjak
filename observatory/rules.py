"""Deterministic, inspectable paper labels. No networking or reference-label input."""
import re
import unicodedata

from .ingest import label_key
from .store import digest, read_json, read_jsonl, write_json, write_jsonl, write_text


def normalize(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    value = re.sub(r'[-‐‑‒–—−_]+', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def load_rules(root):
    book = read_json(root / 'config/paper_rules.json')
    if book is None:
        raise ValueError('Missing config/paper_rules.json')
    for rules in book['rules'].values():
        for rule in rules:
            re.compile(rule[2])
    return book


def predict(paper, book):
    title, abstract = normalize(paper['title']), normalize(paper.get('abstract') or '')
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    contribution = ' '.join(s for s in sentences if re.search(r'\b(?:we (?:introduce|propose|present|develop)|our (?:method|framework|contribution)|this (?:paper|work) (?:introduces|presents|proposes))\b', s))
    scores, evidence = {}, []
    for category, rules in book['rules'].items():
        score = 0
        for rule in rules:
            name, weight, pattern, *scope = rule
            field = scope[0] if scope else 'both'
            # A feature votes once, even if its words appear many times. Title and
            # contribution boosts favor the central contribution over background.
            match = re.search(pattern, title) if field in {'title', 'both'} else None
            multiplier, source = book['title_multiplier'], 'title'
            if not match and field in {'abstract', 'both'}:
                match = re.search(pattern, contribution)
                multiplier, source = book['contribution_multiplier'], 'contribution'
                if not match:
                    match = re.search(pattern, abstract)
                    multiplier, source = 1, 'abstract'
            if match:
                value = round(weight * multiplier, 2)
                score += value
                evidence.append({'category': category, 'rule': name, 'match': match.group()[:180], 'field': source, 'weight': value})
        scores[category] = round(score, 2)
    ranked = sorted(scores, key=lambda c: (-scores[c], c))
    winner, runner_up = ranked[:2]
    margin = round(scores[winner] - scores[runner_up], 2)
    fallback = scores[winner] < book['minimum_score']
    category = 'other' if fallback else winner
    uncertain = fallback or margin < book['uncertain_margin'] or not abstract
    strongest = sorted((e for e in evidence if e['category'] == category and e['weight'] > 0), key=lambda e: -e['weight'])[:3]
    reason = ('No sufficiently strong keyword evidence.' if fallback else
              'Matched ' + ', '.join(e['rule'] for e in strongest) + f'; score {scores[category]:g}, margin {margin:g}.')
    return {'category': category, 'uncertain': uncertain, 'reason': reason,
            'scores': scores, 'margin': margin, 'matched_rules': evidence,
            'runner_up': runner_up, 'fallback': fallback}


def classify_rules(root, config, today, apply=False):
    book = load_rules(root)
    if set(book['rules']) != set(config['categories']):
        raise ValueError('Rule categories must match the configured taxonomy')
    rules_hash = digest(book)
    papers = read_jsonl(root / 'data/papers.jsonl')
    previous = {l['key']: l for l in read_jsonl(root / 'data/labels_rules.jsonl')}
    predictions, recomputed = [], 0
    for paper in papers:
        key = label_key(paper, config)
        old = previous.get(key)
        if old and old.get('rules_hash') == rules_hash:
            predictions.append(old)
            continue
        predictions.append({**predict(paper, book), 'key': key, 'id': paper['id'],
                            'content_hash': paper['content_hash'], 'source': 'keyword_rules',
                            'model': None, 'evidence': 'title_and_full_abstract',
                            'rules_version': book['version'], 'rules_hash': rules_hash,
                            'taxonomy_version': config['taxonomy_version'],
                            'prompt_version': config['prompt_version'], 'labeled_on': today.isoformat()})
        recomputed += 1
    write_jsonl(root / 'data/labels_rules.jsonl', sorted(predictions, key=lambda l: l['key']))
    applied = 0
    if apply:
        active = {l['key']: l for l in read_jsonl(root / 'data/labels.jsonl')}
        for prediction in predictions:
            existing = active.get(prediction['key'])
            # Reviewed historical / HF labels are retained. Only missing records
            # and previous rule-generated labels are managed by this backend.
            if existing is None or existing.get('source') == 'keyword_rules':
                if existing != prediction:
                    active[prediction['key']] = prediction
                    applied += 1
        write_jsonl(root / 'data/labels.jsonl', sorted(active.values(), key=lambda l: l['key']))
    return predictions, {'rule_predictions': len(predictions), 'rules_recomputed': recomputed, 'labels_applied': applied, 'classifier_requests': 0}


def compare(root, config, predictions):
    references = {l['key']: l for l in read_jsonl(root / 'data/labels.jsonl') if l.get('source') != 'keyword_rules'}
    papers = {p['id']: p for p in read_jsonl(root / 'data/papers.jsonl')}
    pairs = [(p, references[p['key']]) for p in predictions if p['key'] in references]
    categories = list(config['categories'])
    matrix = {a: {b: 0 for b in categories} for a in categories}
    matched, confident_matched, confident_total = 0, 0, 0
    disagreements = []
    for prediction, reference in pairs:
        same = prediction['category'] == reference['category']
        matched += same
        matrix[reference['category']][prediction['category']] += 1
        if not reference['uncertain']:
            confident_total += 1
            confident_matched += same
        if not same:
            disagreements.append({'id': prediction['id'], 'title': papers[prediction['id']]['title'],
                                  'reference': reference['category'], 'predicted': prediction['category'],
                                  'reference_uncertain': reference['uncertain'], 'rules_uncertain': prediction['uncertain'],
                                  'margin': prediction['margin'], 'reason': prediction['reason']})
    metrics = {'compared': len(pairs), 'matched': matched, 'agreement': matched / len(pairs) if pairs else None,
               'rules_version': predictions[0]['rules_version'] if predictions else None,
               'rules_hash': predictions[0]['rules_hash'] if predictions else None,
               'corpus_hash': digest(sorted((p['id'], p['content_hash']) for p in predictions)),
               'unflagged_reference_count': confident_total, 'unflagged_reference_matched': confident_matched,
               'rules_uncertain': sum(p['uncertain'] for p, _ in pairs),
               'rules_fallback': sum(p['fallback'] for p, _ in pairs),
               'excluded_without_reference': len(predictions) - len(pairs),
               'confusion_matrix': matrix, 'disagreements': disagreements}
    write_json(root / 'data/label_comparison.json', metrics)
    def pct(a, b):
        return f'{a / b:.1%}' if b else 'N/A'
    lines = ['# Keyword-rule label comparison', '',
             f'Agreement: **{matched}/{len(pairs)} ({pct(matched, len(pairs))})**. On references without an uncertainty flag: **{confident_matched}/{confident_total} ({pct(confident_matched, confident_total)})**.', '',
             'This is agreement with the existing non-rule labels (initially Codex bootstrap), not ground-truth accuracy. Rules were developed with this same historical corpus available, so this is an in-sample diagnostic, not a held-out estimate of future accuracy. No reference labels or paper IDs are used as inputs to prediction. Reference labels are not overwritten.', '',
             f'{metrics["rules_uncertain"]} rule assignments are flagged uncertain; {metrics["rules_fallback"]} use the low-evidence fallback. Scores and margins are heuristic votes, not calibrated probabilities.', '',
             '## Per-category agreement', '', '| Category | Reference count | Rules count | Exact matches | Recall vs reference |', '| --- | ---: | ---: | ---: | ---: |']
    for cat in categories:
        total = sum(matrix[cat].values())
        predicted = sum(matrix[row][cat] for row in categories)
        lines.append(f'| {config["categories"][cat]["name"]} | {total} | {predicted} | {matrix[cat][cat]} | {pct(matrix[cat][cat], total)} |')
    lines += ['', '## Confusion matrix', '', 'Rows = reference; columns = rules.', '', '| Reference / rules | ' + ' | '.join(categories) + ' |', '| --- | ' + ' | '.join(['---:'] * len(categories)) + ' |']
    for cat in categories:
        lines.append('| ' + cat + ' | ' + ' | '.join(str(matrix[cat][other]) for other in categories) + ' |')
    lines += ['', '## Disagreements', '', '| Paper | Reference | Rules | Reference uncertain? | Rules uncertain? | Evidence |', '| --- | --- | --- | --- | --- | --- |']
    for row in disagreements:
        title = row['title'].replace('|', '\\|').replace('[', '\\[').replace(']', '\\]').replace('<', '&lt;').replace('>', '&gt;').replace('\n', ' ')
        lines.append(f'| [{title}](https://huggingface.co/papers/{row["id"]}) | {row["reference"]} | {row["predicted"]} | {row["reference_uncertain"]} | {row["rules_uncertain"]} | {row["reason"]} |')
    lines += ['', 'Rules: [paper_rules.json](../config/paper_rules.json). Full matched phrases, scores, and provenance: [labels_rules.jsonl](../data/labels_rules.jsonl). Machine-readable metrics: [label_comparison.json](../data/label_comparison.json).']
    write_text(root / 'reports/label-comparison.md', '\n'.join(lines) + '\n')
    return {k: v for k, v in metrics.items() if k not in {'confusion_matrix', 'disagreements'}}

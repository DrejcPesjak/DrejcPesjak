"""Incrementally patch persistent SVGs; templates are created only on first use.

Named slots own axes/annotations; keyed record blocks own model points and paper
days. Unchanged blocks, custom artwork and the scaffold remain byte-for-byte.
Atomic file replacement is a filesystem write, not a full SVG regeneration.
"""
import html
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from .ingest import label_key
from .store import completed_dates, digest, read_json, read_jsonl, write_json, write_text

INK, MUTED, FAINT = '#e6edf7', '#98a6bc', '#52617b'


def esc(value):
    return html.escape(str(value), quote=True)


def text(x, y, value, size=12, color=MUTED, **attrs):
    extra = ' '.join(f'{key.replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" {extra}>{esc(value)}</text>'


def compact(n):
    for scale, unit in [(1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'k')]:
        if n >= scale:
            return f'{n / scale:.1f}'.rstrip('0').rstrip('.') + unit
    return str(int(n))


def slot(name, content=''):
    return f'<!--slot:{name}-->{content}<!--/slot:{name}-->'


def replace_slot(svg, name, content):
    start, end = f'<!--slot:{name}-->', f'<!--/slot:{name}-->'
    if svg.count(start) != 1 or svg.count(end) != 1:
        raise ValueError(f'Missing or duplicate SVG slot: {name}')
    left, rest = svg.split(start)
    _, right = rest.split(end)
    return left + start + content + end + right


def reconcile_records(svg, desired):
    """Keep existing keyed blocks in place, patch changed ones, append new ones."""
    start, end = '<!--slot:records-->', '<!--/slot:records-->'
    body = svg.split(start)[1].split(end)[0]
    pattern = re.compile(r'<!--record:([a-f0-9]+)-->(.*?)<!--/record:\1-->', re.S)
    found = {match.group(1): match.group(0) for match in pattern.finditer(body)}
    changes = {'added': 0, 'updated': 0, 'removed': 0}
    for key, old in found.items():
        if key not in desired:
            body = body.replace(old, '')
            changes['removed'] += 1
        else:
            new = f'<!--record:{key}-->{desired[key]}<!--/record:{key}-->'
            if new != old:
                body = body.replace(old, new)
                changes['updated'] += 1
    for key, content in desired.items():
        if key not in found:
            body += f'<!--record:{key}-->{content}<!--/record:{key}-->'
            changes['added'] += 1
    return replace_slot(svg, 'records', body), changes


def scaffold(kind):
    title = 'The model landscape' if kind == 'models' else 'A week in research'
    subtitle = ('Which language models attract use — and how large are they?' if kind == 'models'
                else 'One paper. One colored mark. Seven completed days.')
    eyebrow = '01 / MODEL ADOPTION' if kind == 'models' else '02 / RESEARCH RAIN'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="660" viewBox="0 0 1000 660" role="img" aria-labelledby="title desc">
<title id="title">{slot('accessible-title', title)}</title>
<desc id="desc">{slot('description')}</desc>
<defs>
  <linearGradient id="surface" x2="1" y2="1"><stop stop-color="#111d30"/><stop offset="1" stop-color="#090f1b"/></linearGradient>
  <radialGradient id="aura"><stop stop-color="#7060cd" stop-opacity=".16"/><stop offset="1" stop-color="#7060cd" stop-opacity="0"/></radialGradient>
  <clipPath id="plot-clip"><rect x="84" y="190" width="850" height="314"/></clipPath>
</defs>
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
.mono {{ font-family: ui-monospace, 'SFMono-Regular', Consolas, monospace; }}
.dot {{ animation: appear .8s cubic-bezier(.2,.8,.2,1) both; transform-box: fill-box; transform-origin: center; }}
.paper {{ animation: rain 1.05s cubic-bezier(.2,.75,.25,1) both; }}
@keyframes appear {{ from {{ opacity:0; transform:scale(.2); }} to {{ opacity:1; transform:scale(1); }} }}
@keyframes rain {{ from {{ opacity:0; transform:translateY(-320px); }} 18% {{ opacity:1; }} to {{ opacity:1; transform:translateY(0); }} }}
@media (prefers-reduced-motion: reduce) {{ .dot, .paper {{ animation:none !important; }} }}
</style>
<rect x=".5" y=".5" width="999" height="659" rx="22" fill="url(#surface)" stroke="#25334a"/>
<ellipse cx="840" cy="60" rx="340" ry="190" fill="url(#aura)"/>
<path d="M34 39h20" stroke="#a78bfa" stroke-width="3" stroke-linecap="round"/>
{text(66, 43, eyebrow, 11, '#bdacd9', letter_spacing='2', font_weight='600')}
{slot('heading', text(34, 87, title, 31, INK, font_weight='650', letter_spacing='-.8'))}
{slot('subtitle', text(34, 115, subtitle, 14))}
{slot('meta')}
<path d="M34 141H966" stroke="#263248"/>
{slot('stats')}
{slot('axes')}
{slot('records')}
{slot('annotations')}
{slot('legend')}
<path d="M34 609H966" stroke="#263248"/>
{slot('footer')}
</svg>
'''


def save_chart(root, kind, slots, records):
    path = root / 'assets' / (kind + '.svg')
    svg = path.read_text() if path.exists() else scaffold(kind)
    for name, content in slots.items():
        svg = replace_slot(svg, name, content)
    svg, changes = reconcile_records(svg, records)
    ET.fromstring(svg)
    if len(svg.encode()) > 200_000:
        raise ValueError(f'{kind} SVG exceeds 200 KB budget')
    changed = write_text(path, svg)
    static = svg.replace('</style>', '\n.dot, .paper { animation:none !important; }\n</style>')
    write_text(root / 'assets' / (kind + '-static.svg'), static)
    return {'changed': changed, **changes}


def family_color(family, config):
    return config['families'].get(family, {}).get('color', '#7c899e')


def model_chart(root, config, today):
    snapshot = read_json(root / 'data/models.json', {})
    models = snapshot.get('models', [])
    state_path = root / 'data/chart_state.json'
    state = read_json(state_path, {})
    # Domains only expand: routine additions do not shift all existing marks.
    bounds = state.get('model_bounds', [8, 12, 3, 8])
    if models:
        bounds = [min(bounds[0], math.floor(math.log10(min(m['params'] for m in models)))),
                  max(bounds[1], math.ceil(math.log10(max(m['params'] for m in models)))),
                  min(bounds[2], math.floor(math.log10(min(m['downloads'] for m in models)))),
                  max(bounds[3], math.ceil(math.log10(max(m['downloads'] for m in models))))]
    x0, x1, y0, y1 = bounds
    x = lambda n: 88 + (math.log10(n) - x0) / (x1 - x0) * 834
    y = lambda n: 489 - (math.log10(n) - y0) / (y1 - y0) * 278
    axes = text(88, 192, 'DOWNLOADS / LAST 30 DAYS', 10, MUTED, letter_spacing='1.2')
    for exponent in range(y0, y1 + 1):
        yy = y(10 ** exponent)
        axes += f'<path d="M88 {yy:.2f}H922" stroke="#29364b" stroke-dasharray="2 5"/>'
        axes += text(76, round(yy + 4, 2), compact(10 ** exponent), 11, text_anchor='end', **{'class': 'mono'})
    for exponent in range(x0, x1 + 1):
        xx = x(10 ** exponent)
        axes += f'<path d="M{xx:.2f} 207V489" stroke="#263247" stroke-dasharray="2 5"/>'
        axes += text(round(xx, 2), 512, compact(10 ** exponent), 11, text_anchor='middle', **{'class': 'mono'})
    axes += text(505, 537, 'TOTAL PARAMETERS · LOGARITHMIC AXES', 10, MUTED, text_anchor='middle', letter_spacing='1.2')
    records = {}
    for model in models:
        xx, yy = x(model['params']), y(model['downloads'])
        color = family_color(model['family'], config)
        key = digest(model['id'])[:20]
        delay = int(key[:4], 16) % 700 / 1000
        content = f'<g id="model-{key}" transform="translate({xx:.2f} {yy:.2f})"><title>{esc(model["id"])} · {compact(model["params"])} parameters · {compact(model["downloads"])} downloads</title>'
        content += f'<g class="dot" style="animation-delay:{delay}s"><circle r="11" fill="{color}" opacity=".09"/><circle r="5.2" fill="{color}" stroke="#111b2d" stroke-width="1.5"/></g></g>'
        records[key] = content
    # Label high-use models and family representatives; avoid point/label overlap.
    candidates = sorted(models, key=lambda m: -m['downloads'])
    chosen = candidates[:3]
    for family in config['families']:
        representative = next((m for m in candidates if m['family'] == family), None)
        if representative and representative not in chosen:
            chosen.append(representative)
    annotations, boxes = '', []
    for model in chosen[:8]:
        xx, yy = x(model['params']), y(model['downloads'])
        name = model['id'].split('/')[-1]
        name = name if len(name) <= 26 else name[:24] + '…'
        width = len(name) * 6.5 + 12
        for dx, dy in [(12, -13), (12, 23), (-width - 12, -13), (-width - 12, 23), (12, -37)]:
            bx, by = max(90, min(920 - width, xx + dx)), yy + dy - 13
            box = (bx, by, bx + width, by + 20)
            if by < 198 or by + 20 > 490 or any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in boxes):
                continue
            if any(box[0] - 5 < x(m['params']) < box[2] + 5 and box[1] - 5 < y(m['downloads']) < box[3] + 5 for m in models):
                continue
            boxes.append(box)
            annotations += f'<path d="M{xx:.2f} {yy:.2f}L{bx + (0 if dx > 0 else width):.2f} {by + 10:.2f}" stroke="{family_color(model["family"], config)}" opacity=".45"/>'
            annotations += f'<rect x="{bx:.2f}" y="{by:.2f}" width="{width}" height="20" rx="4" fill="#101a2b" stroke="#2b3850"/>'
            annotations += text(round(bx + 6, 2), round(by + 14, 2), name, 11, INK)
            break
    legend = ''
    for index, family in enumerate([*config['families'], 'Other']):
        xx = 50 + index * 133
        legend += f'<circle cx="{xx}" cy="575" r="4" fill="{family_color(family, config)}"/>'
        legend += text(xx + 10, 579, family, 11)
    observed = snapshot.get('date', 'not collected')
    status = ('SNAPSHOT ' + observed) if observed == today.isoformat() else ('LAST DATA ' + observed)
    stats = text(34, 170, f'{len(models):02d} MODELS', 12, INK, font_weight='600', letter_spacing='1')
    stats += text(966, 170, 'COLOR = PUBLISHER  /  SIZE = TOTAL WEIGHTS', 10, text_anchor='end', letter_spacing='.6')
    if not models:
        annotations += text(500, 350, 'Waiting for a valid model snapshot', 18, text_anchor='middle')
    result = save_chart(root, 'models', {
        'accessible-title': 'The model landscape',
        'heading': text(34, 87, 'The model landscape', 31, INK, font_weight='650', letter_spacing='-.8'),
        'subtitle': text(34, 115, 'Which language models attract use — and how large are they?', 14),
        'description': esc(f'{len(models)} selected language models. Total parameters versus 30-day HF downloads. Colors identify publishers. Data: {observed}.'),
        'meta': text(966, 43, status, 10, text_anchor='end', **{'class': 'mono'}),
        'stats': stats, 'axes': axes, 'annotations': annotations, 'legend': legend,
        'footer': text(34, 635, 'HUGGING FACE', 10, '#a78bfa', letter_spacing='1.5') + text(195, 635, 'A selected sample · downloads indicate activity, not quality or unique users.', 11),
    }, records)
    state['model_bounds'] = bounds
    write_json(state_path, state)
    return result


def paper_data(root, config, today):
    days = read_json(root / 'data/days.json', {})
    papers = {p['id']: p for p in read_jsonl(root / 'data/papers.jsonl')}
    labels = {l['key']: l for l in read_jsonl(root / 'data/labels.jsonl')}
    result = []
    for day in completed_dates(today, config['window_days']):
        membership = days.get(day, {})
        rows = []
        for pid in membership.get('ids', []):
            p = papers[pid]
            label = labels.get(label_key(p, config))
            rows.append({**p, 'category': label['category'] if label else 'pending',
                         'label_source': label.get('source', 'hf_inference') if label else 'pending',
                         'uncertain': label.get('uncertain', False) if label else True})
        result.append({'date': day, 'complete': membership.get('complete', False), 'papers': rows})
    return result


def paper_chart(root, config, today):
    days = paper_data(root, config, today)
    categories = {**config['categories'], 'pending': {'name': 'Pending labels', 'color': '#475569'}}
    maximum = max([len(d['papers']) for d in days] + [1])
    state_path = root / 'data/chart_state.json'
    state = read_json(state_path, {})
    ymax = max(state.get('paper_ymax', 10), math.ceil(maximum / 10) * 10)
    unit = 275 / ymax
    baseline = 489
    axes = text(88, 192, 'DAILY PAPER APPEARANCES', 10, letter_spacing='1.2')
    step = max(5, math.ceil(ymax / 5 / 5) * 5)
    for count in range(0, ymax + 1, step):
        yy = baseline - count * unit
        axes += f'<path d="M88 {yy:.2f}H934" stroke="#29364b" stroke-dasharray="2 5"/>'
        axes += text(76, round(yy + 4, 2), count, 11, text_anchor='end', **{'class': 'mono'})
    width = 828 / len(days)
    bar_width = min(78, width * .66)
    records = {}
    count_all, pending_count = 0, 0
    order = list(categories)
    aggregated = sum(len(d['papers']) for d in days) > 450
    for index, day in enumerate(days):
        xx = 94 + index * width + width / 2
        rows = sorted(day['papers'], key=lambda p: (order.index(p['category']), p['id']))
        count_all += len(rows)
        pending_count += sum(p['category'] == 'pending' for p in rows)
        key = digest(day['date'])[:20]
        body = f'<g id="day-{day["date"]}" transform="translate({xx:.2f} 0)">'
        if not day['complete']:
            body += f'<rect x="{-bar_width / 2:.2f}" y="227" width="{bar_width:.2f}" height="262" rx="6" fill="#192234" opacity=".5" stroke="#64748b" stroke-dasharray="3 6"/>'
            body += text(0, 371, 'NO DATA', 9, text_anchor='middle')
        else:
            segments = Counter(p['category'] for p in rows)
            offset = 0
            for ci, category in enumerate(order):
                group = [p for p in rows if p['category'] == category]
                if not group:
                    continue
                if aggregated:
                    group = [None]
                for item in group:
                    count = segments[category] if item is None else 1
                    height = unit * count
                    yy = baseline - (offset + count) * unit
                    delay = .12 * index + (offset / max(1, maximum)) * 2.1
                    title = f'{count} {categories[category]["name"]} papers' if item is None else item['title']
                    pid = digest([day['date'], item['id'] if item else category])[:16]
                    records[pid] = f'<g clip-path="url(#plot-clip)"><g transform="translate({xx:.2f} 0)"><rect id="paper-{pid}" class="paper" x="{-bar_width / 2:.2f}" y="{yy:.2f}" width="{bar_width:.2f}" height="{max(.25, height - min(.8, height * .15)):.2f}" rx="1.2" fill="{categories[category]["color"]}" style="animation-delay:{delay:.3f}s"><title>{esc(title)}</title></rect></g></g>'
                    offset += count
            body += text(0, round(baseline - len(rows) * unit - 12, 2), len(rows), 15, INK, text_anchor='middle', **{'class': 'mono'})
        from datetime import date
        dt = date.fromisoformat(day['date'])
        body += text(0, 516, dt.strftime('%a').upper(), 10, text_anchor='middle', letter_spacing='1')
        if len(days) <= 10 or index % 4 == 0 or index == len(days) - 1:
            body += text(0, 534, dt.strftime('%d %b'), 10, FAINT, text_anchor='middle')
        body += '</g>'
        records[key] = body
    legend = ''
    visible_categories = list(config['categories']) + (['pending'] if pending_count else [])
    for index, category in enumerate(visible_categories):
        xx, yy = 40 + (index % 4) * 237, 563 + (index // 4) * 19
        legend += f'<rect x="{xx}" y="{yy - 7}" width="7" height="7" rx="2" fill="{categories[category]["color"]}"/>'
        legend += text(xx + 14, yy, categories[category]['name'], 10)
    complete = sum(d['complete'] for d in days)
    unique = len({p['id'] for d in days for p in d['papers']})
    stats = text(34, 170, f'{count_all} APPEARANCES', 12, INK, font_weight='600', letter_spacing='1')
    stats += text(966, 170, f'{unique} UNIQUE PAPERS  /  {complete} OF {len(days)} DAYS COLLECTED' + (f'  /  {pending_count} UNLABELED' if pending_count else ''), 10, text_anchor='end')
    heading = 'A week in research' if len(days) == 7 else f'{len(days)} days in research'
    result = save_chart(root, 'papers', {
        'accessible-title': esc(heading),
        'heading': text(34, 87, heading, 31, INK, font_weight='650', letter_spacing='-.8'),
        'subtitle': text(34, 115, f'One paper. One colored mark. {len(days)} completed days.', 14),
        'description': esc(f'{count_all} paper appearances across {complete} collected days, {days[0]["date"]} to {days[-1]["date"]}. Colors identify primary research categories. {pending_count} labels pending. Missing days are not zero.'),
        'meta': text(966, 43, f'{days[0]["date"]} — {days[-1]["date"]} UTC', 10, text_anchor='end', **{'class': 'mono'}),
        'stats': stats, 'axes': axes, 'annotations': '', 'legend': legend,
        'footer': text(34, 635, 'HUGGING FACE', 10, '#a78bfa', letter_spacing='1.5') + text(195, 635, 'Daily-list dates · one primary category per paper · ' + ('marks aggregate category counts.' if aggregated else 'one falling mark = one paper.'), 11),
    }, records)
    state['paper_ymax'] = ymax
    write_json(state_path, state)
    return result


def md(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('[', '\\[').replace(']', '\\]').replace('<', '&lt;').replace('>', '&gt;')


def reports(root, config, today):
    snapshot = read_json(root / 'data/models.json', {})
    lines = ['# Model observations', '', f'Snapshot: {snapshot.get("date", "not collected")} UTC. Downloads cover the preceding 30 days.', '',
             'Colors identify the publishing account, not inferred ancestry. VLMs that generate text are eligible. Size is HF’s structured total parameter count; MoE totals include all experts. This is a bounded sample, not a quality ranking.', '',
             '| Model | Publisher | Total parameters | 30-day downloads | Task |', '| --- | --- | ---: | ---: | --- |']
    for m in sorted(snapshot.get('models', []), key=lambda m: -m['downloads']):
        url = 'https://huggingface.co/' + quote(m['id'], safe='/')
        lines.append(f'| [{md(m["id"])}]({url}) | {md(m["family"])} | {m["params"]:,} | {m["downloads"]:,} | {m["task"]} |')
    lines += ['', '## Excluded candidates', '', '| Model | Reason |', '| --- | --- |']
    for item in snapshot.get('excluded', []):
        lines.append(f'| {md(item["id"])} | {item["reason"]} |')
    write_text(root / 'reports/models.md', '\n'.join(lines) + '\n')
    lines = ['# Daily research papers', '', 'Dates are HF daily-list dates, not original publication dates. Repeat appearances on different days are retained; arXiv versions and same-day duplicates are collapsed. Labels are machine classifications of title/abstract evidence; “uncertain” flags deserve review.', '',
             'Label sources distinguish Codex bootstrap review, local keyword rules, and optional HF inference. Rule scores are heuristic; they are not calibrated probabilities. [Bootstrap audit](labeling-bootstrap.md) · [Rule comparison](label-comparison.md).', '']
    for day in reversed(paper_data(root, config, today)):
        lines += [f'## {day["date"]}', '', ('Complete daily list.' if day['complete'] else '**Missing data — not a zero-paper day.**'), '', '| Paper | Primary category | Review | Label source |', '| --- | --- | --- | --- |']
        for p in day['papers']:
            category = config['categories'].get(p['category'], {}).get('name', 'Pending')
            source = {'assistant_bootstrap': 'Codex bootstrap', 'keyword_rules': 'Keyword rules', 'hf_inference': 'HF inference', 'pending': 'Pending'}.get(p['label_source'], p['label_source'])
            lines.append(f'| [{md(p["title"])}]({p["url"]}) | {category} | {"uncertain" if p["uncertain"] else ""} | {md(source)} |')
        lines.append('')
    write_text(root / 'reports/papers.md', '\n'.join(lines) + '\n')


def render(root, config, today):
    # Each SVG is validated before its atomic replacement; data is checkpointed
    # separately so a failed render can be rerun without fetching or relabeling.
    result = {'models': model_chart(root, config, today), 'papers': paper_chart(root, config, today)}
    reports(root, config, today)
    return result

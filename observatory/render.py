"""Incrementally patch persistent SVGs; templates are created only on first use.

Named slots own axes/annotations; keyed record blocks own model points and paper
days. Unchanged blocks, custom artwork and the scaffold remain byte-for-byte.
Atomic file replacement is a filesystem write, not a full SVG regeneration.
"""
import base64
import html
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from .ingest import label_key
from .store import completed_dates, digest, read_json, read_jsonl, write_json, write_text

INK, MUTED, FAINT = '#11191d', '#252936', '#49494c'
PAPER = '#f8f2e6'
# Pigment versions of the configured colors, shared by marks and legends.
PIGMENTS = {'#a78bfa': '#b398f0', '#60a5fa': '#6ab6ef', '#fbbf24': '#ffd45c',
            '#2dd4bf': '#4bd0bb', '#fb923c': '#ffb45e', '#f472b6': '#f49dc6',
            '#c4b5fd': '#c5b0ef', '#7c899e': '#a2a7a5', '#475569': '#888d91'}


def pigment(color):
    return PIGMENTS.get(color, color)


def pencil_line(x1, y1, x2, y2, seed, color=INK, width=.9, dashed=False):
    # Short, seeded pencil movements keep the endpoints and measured axes fixed.
    dx, dy = x2 - x1, y2 - y1
    steps = max(2, math.ceil(math.hypot(dx, dy) / 18))
    points = [f'M{x1:.2f} {y1:.2f}']
    for i in range(1, steps):
        bend = (int(digest([str(seed), i])[:4], 16) % 21 - 10) / 30
        points.append(f'L{x1 + dx*i/steps + (bend if dy else 0):.2f} '
                      f'{y1 + dy*i/steps + (bend if dx else 0):.2f}')
    points.append(f'L{x2:.2f} {y2:.2f}')
    return (f'<path d="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" '
            f'stroke-dasharray="{"3 4" if dashed else "none"}"/>')


def ink_dot(color, seed, radius=5.2):
    tilt = int(seed[:4], 16) % 50 - 25
    return (f'<g transform="rotate({tilt})"><ellipse rx="{radius + 1.3}" ry="{radius - .3}" '
            f'fill="{color}" fill-opacity=".25"/>'
            f'<path d="M{-radius:.2f} -.8 C-5 -7 5 -6 {radius:.2f} -.5 '
            f'C7 5 -4 7 {-radius:.2f} -.8Z" fill="{color}" fill-opacity=".95" '
            f'stroke="{INK}" stroke-width="1.1" stroke-linejoin="round"/></g>')


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
    eyebrow = '01 / MODEL ADOPTION' if kind == 'models' else '02 / RESEARCH RAIN'
    washes = ('<g clip-path="url(#card-clip)"><ellipse cx="850" cy="56" rx="300" ry="145" fill="#b9d8d3" opacity=".16"/>'
              '<ellipse cx="125" cy="565" rx="155" ry="90" fill="#e8c9aa" opacity=".12"/></g>' if kind == 'models' else
              '<ellipse cx="910" cy="103" rx="100" ry="62" fill="#c3d1b5" opacity=".09"/>\n'
              '<ellipse cx="704" cy="126" rx="61" ry="38" fill="#dfc69b" opacity=".08"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="660" viewBox="0 0 1000 660" role="img" aria-labelledby="title desc">
<title id="title">{slot('accessible-title', title)}</title>
<desc id="desc">{slot('description')}</desc>
<defs>
  <filter id="paper-grain" x="0" y="0" width="100%" height="100%">
    <feTurbulence type="fractalNoise" baseFrequency=".72" numOctaves="3" seed="23"/>
    <feColorMatrix type="saturate" values="0"/>
    <feComponentTransfer><feFuncA type="linear" slope=".09"/></feComponentTransfer>
    <feBlend in="SourceGraphic" mode="multiply"/>
  </filter>
  <pattern id="pigment-grain" width="37" height="19" patternUnits="userSpaceOnUse"><path d="M2 3l13 .3 M23 12l10 -.4 M7 17l4 .2" stroke="#fffaf0" stroke-opacity=".25" stroke-width=".5"/></pattern>
  <clipPath id="card-clip"><rect x="6" y="6" width="988" height="648" rx="18"/></clipPath>
  <clipPath id="plot-clip"><rect x="84" y="190" width="850" height="314"/></clipPath>
</defs>
<style>
{slot('model-font') if kind == 'models' else ''}
:root {{ color-scheme: light; }}
text {{ font-family: 'DejaVu Sans Mono', 'Courier New', monospace; }}
.heading {{ font-family: Georgia, 'Times New Roman', serif; font-style:normal; }}
.subtitle {{ font-family: Georgia, 'Times New Roman', serif; }}
.mono {{ font-family: ui-monospace, 'SFMono-Regular', Consolas, monospace; }}
.annotations text {{ font-family: 'Segoe Print', 'Bradley Hand', 'Comic Sans MS', cursive; }}
.dot {{ animation: appear .8s cubic-bezier(.2,.8,.2,1) both; transform-box: fill-box; transform-origin: center; }}
.paper {{ animation: rain 1.05s cubic-bezier(.2,.75,.25,1) both; }}
@keyframes appear {{ from {{ opacity:0; transform:scale(.2); }} to {{ opacity:1; transform:scale(1); }} }}
@keyframes rain {{ from {{ opacity:0; transform:translateY(-320px); }} 18% {{ opacity:1; }} to {{ opacity:1; transform:translateY(0); }} }}
@media (prefers-reduced-motion: reduce) {{ .dot, .paper {{ animation:none !important; }} }}
</style>
<rect x="4" y="4" width="992" height="652" rx="18" fill="{PAPER}" filter="url(#paper-grain)"/>
{washes}
<path d="M24 6 C265 4 675 7 976 5 Q993 6 994 24 C995 208 993 453 994 637 Q993 653 977 654 C662 652 307 655 24 654 Q6 653 6 636 C5 432 7 203 6 24 Q7 7 24 6Z" fill="none" stroke="{INK}" stroke-width="1.45" stroke-linecap="round"/>
<path d="M36 32 Q47 31.5 62 32" fill="none" stroke="#ad8ded" stroke-width="4" stroke-linecap="round"/>
{text(75, 37, eyebrow, 12, INK, letter_spacing='2')}
{slot('heading')}
{slot('subtitle')}
{slot('meta')}
{pencil_line(34, 141, 966, 141, 'header')}
{slot('stats')}
<g class="axes">{slot('axes')}</g>
{slot('records')}
<g class="annotations">{slot('annotations')}</g>
<g class="legend">{slot('legend')}</g>
{pencil_line(34, 609, 966, 609, 'footer')}
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
    return pigment(config['families'].get(family, {}).get('color', '#7c899e'))


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
    axes = text(88, 192, 'DOWNLOADS / LAST 30 DAYS', 12, MUTED, letter_spacing='1.2')
    for exponent in range(y0, y1 + 1):
        yy = y(10 ** exponent)
        axes += pencil_line(88, yy, 922, yy, f'model-y-{exponent}', '#bdb7ab', dashed=True)
        axes += text(76, round(yy + 4, 2), compact(10 ** exponent), 12, text_anchor='end', **{'class': 'mono'})
    for exponent in range(x0, x1 + 1):
        xx = x(10 ** exponent)
        axes += pencil_line(xx, 207, xx, 489, f'model-x-{exponent}', '#dad3c6', dashed=True)
        axes += text(round(xx, 2), 512, compact(10 ** exponent), 12, text_anchor='middle', **{'class': 'mono'})
    axes += text(505, 537, 'TOTAL PARAMETERS · LOGARITHMIC AXES', 10, MUTED, text_anchor='middle', letter_spacing='1.2')
    axes += pencil_line(88, 202, 88, 489, 'model-left', width=1.4)
    axes += pencil_line(88, 489, 932, 489, 'model-bottom', width=1.4)
    records = {}
    for model in models:
        xx, yy = x(model['params']), y(model['downloads'])
        color = family_color(model['family'], config)
        key = digest(model['id'])[:20]
        delay = int(key[:4], 16) % 700 / 1000
        content = f'<g id="model-{key}" transform="translate({xx:.2f} {yy:.2f})"><title>{esc(model["id"])} · {compact(model["params"])} parameters · {compact(model["downloads"])} downloads</title>'
        content += f'<g class="dot" style="animation-delay:{delay}s">{ink_dot(color, key)}</g></g>'
        records[key] = content
    # Label high-use models and family representatives; avoid point/label overlap.
    candidates = sorted(models, key=lambda m: -m['downloads'])
    chosen = candidates[:3]
    for family in config['families']:
        representative = next((m for m in candidates if m['family'] == family), None)
        if representative and representative not in chosen:
            chosen.append(representative)
    label_metrics = read_json(Path(__file__).resolve().parents[1] / 'assets/fonts/chilanka-metrics.json')
    annotations, boxes = '', []
    for model in chosen[:8]:
        xx, yy = x(model['params']), y(model['downloads'])
        name = model['id'].split('/')[-1]
        name = name if len(name) <= 26 else name[:24] + '…'
        width = math.ceil(sum(label_metrics.get(char, 1) for char in name) * 16) + 20
        for dx, dy in [(14, -19), (14, 32), (-width - 14, -19), (-width - 14, 32), (14, -48), (14, 61)]:
            bx, by = max(90, min(920 - width, xx + dx)), yy + dy - 17
            box = (bx, by, bx + width, by + 27)
            if by < 198 or by + 27 > 490 or any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in boxes):
                continue
            if any(box[0] - 5 < x(m['params']) < box[2] + 5 and box[1] - 5 < y(m['downloads']) < box[3] + 5 for m in models):
                continue
            boxes.append(box)
            annotations += f'<path d="M{xx:.2f} {yy:.2f}L{bx + (0 if dx > 0 else width):.2f} {by + 13:.2f}" stroke="{family_color(model["family"], config)}" opacity=".45"/>'
            annotations += f'<rect x="{bx:.2f}" y="{by:.2f}" width="{width}" height="27" rx="2" fill="#fcf8ef" stroke="#726e65"/>'
            annotations += text(round(bx + 9, 2), round(by + 19, 2), name, 16, INK)
            break
    legend = ''
    for index, family in enumerate([*config['families'], 'Other']):
        xx = 50 + index * 133
        legend += f'<g transform="translate({xx} 575)">{ink_dot(family_color(family, config), digest(family))}</g>'
        legend += text(xx + 10, 579, family, 12)
    observed = snapshot.get('date', 'not collected')
    status = ('SNAPSHOT ' + observed) if observed == today.isoformat() else ('LAST DATA ' + observed)
    stats = text(36, 173, f'{len(models):02d} MODELS', 20, INK, font_weight='600', letter_spacing='1')
    stats += text(966, 170, 'COLOR = PUBLISHER  /  X = TOTAL PARAMETERS', 10, text_anchor='end', letter_spacing='.6')
    if not models:
        annotations += text(500, 350, 'Waiting for a valid model snapshot', 18, text_anchor='middle')
    font_path = Path(__file__).resolve().parents[1] / 'assets/fonts/chilanka-latin.woff2'
    font_data = base64.b64encode(font_path.read_bytes()).decode('ascii')
    result = save_chart(root, 'models', {
        'model-font': (f"@font-face {{ font-family: 'Model Hand'; src: url(data:font/woff2;base64,{font_data}) format('woff2'); }}"
                       "\n.annotations text { font-family: 'Model Hand', Chilanka, cursive !important; }"),
        'accessible-title': 'The model landscape',
        'heading': text(35, 90, 'The model landscape', 52, INK, font_weight='700', letter_spacing='-1', **{'class': 'heading'}),
        'subtitle': text(36, 121, 'Which language models attract use — and how large are they?', 20, INK, **{'class': 'subtitle'}),
        'description': esc(f'{len(models)} selected language models. Total parameters versus 30-day HF downloads. Colors identify publishers. Data: {observed}.'),
        'meta': text(960, 37, status, 12, text_anchor='end', **{'class': 'mono'}),
        'stats': stats, 'axes': axes, 'annotations': annotations, 'legend': legend,
        'footer': text(34, 635, 'HUGGING FACE', 12, '#30315d', letter_spacing='1.5') + text(195, 635, 'A selected sample · downloads indicate activity, not quality or unique users.', 11),
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
    categories = {key: {**value, 'color': pigment(value['color'])}
                  for key, value in {**config['categories'], 'pending': {'name': 'Pending labels', 'color': '#475569'}}.items()}
    maximum = max([len(d['papers']) for d in days] + [1])
    state_path = root / 'data/chart_state.json'
    state = read_json(state_path, {})
    ymax = max(state.get('paper_ymax', 10), math.ceil(maximum / 10) * 10)
    unit = 275 / ymax
    baseline = 489
    axes = text(88, 192, 'DAILY PAPER APPEARANCES', 12, letter_spacing='1.2')
    step = max(5, math.ceil(ymax / 5 / 5) * 5)
    for count in range(0, ymax + 1, step):
        yy = baseline - count * unit
        axes += pencil_line(88, yy, 934, yy, f'paper-y-{count}', '#bdb7ab', dashed=True)
        axes += text(76, round(yy + 4, 2), count, 13, text_anchor='end', **{'class': 'mono'})
    axes += pencil_line(88, 204, 88, 489, 'paper-left', width=1.4)
    axes += pencil_line(88, 489, 946, 489, 'paper-bottom', width=1.4)
    width = 828 / len(days)
    bar_width = min(86, width * .72)
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
            body += f'<rect x="{-bar_width / 2:.2f}" y="227" width="{bar_width:.2f}" height="262" rx="6" fill="#eee6d8" opacity=".7" stroke="#a39987" stroke-dasharray="3 6"/>'
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
                    # Wavering horizontal edges suggest a pencil stroke while the
                    # vertical extent and stacking still encode the exact count.
                    half = bar_width / 2
                    bottom = yy + height
                    wobble = min(.65, height * .12) * (1 if int(pid[:2], 16) % 2 else -1)
                    outline = (f'M{-half:.2f} {yy:.2f} Q0 {yy + wobble:.2f} {half:.2f} {yy:.2f} '
                               f'L{half:.2f} {bottom:.2f} Q0 {bottom - wobble:.2f} {-half:.2f} {bottom:.2f}Z')
                    records[pid] = (f'<g clip-path="url(#plot-clip)"><g transform="translate({xx:.2f} 0)">'
                                    f'<g id="paper-{pid}" class="paper" style="animation-delay:{delay:.3f}s">'
                                    f'<title>{esc(title)}</title><path d="{outline}" fill="{categories[category]["color"]}" fill-opacity=".96" stroke="{INK}" stroke-width=".55" stroke-linejoin="round"/>'
                                    f'<path d="{outline}" fill="url(#pigment-grain)"/></g></g></g>')
                    offset += count
            if not rows:
                body += pencil_line(-bar_width / 2, baseline, bar_width / 2, baseline, key, width=1.5)
            body += text(0, round(baseline - len(rows) * unit - 12, 2), len(rows), 18, INK, text_anchor='middle', **{'class': 'mono'})
        from datetime import date
        dt = date.fromisoformat(day['date'])
        body += text(0, 516, dt.strftime('%a').upper(), 12, text_anchor='middle', letter_spacing='1')
        if len(days) <= 10 or index % 4 == 0 or index == len(days) - 1:
            body += text(0, 534, dt.strftime('%d %b'), 12, FAINT, text_anchor='middle')
        body += '</g>'
        records[key] = body
    legend = ''
    visible_categories = list(config['categories']) + (['pending'] if pending_count else [])
    for index, category in enumerate(visible_categories):
        xx, yy = 54 + (index % 4) * 245, 563 + (index // 4) * 25
        legend += f'<g transform="translate({xx} {yy - 4}) scale(1.35)">{ink_dot(categories[category]["color"], digest(category))}</g>'
        legend += text(xx + 19, yy, categories[category]['name'], 12)
    complete = sum(d['complete'] for d in days)
    unique = len({p['id'] for d in days for p in d['papers']})
    stats = text(36, 173, f'{count_all} APPEARANCES', 20, INK, font_weight='600', letter_spacing='1')
    stats += text(966, 170, f'{unique} UNIQUE PAPERS  /  {complete} OF {len(days)} DAYS COLLECTED' + (f'  /  {pending_count} UNLABELED' if pending_count else ''), 12, text_anchor='end')
    heading = 'A week in research' if len(days) == 7 else f'{len(days)} days in research'
    result = save_chart(root, 'papers', {
        'accessible-title': esc(heading),
        'heading': text(35, 90, heading, 52, INK, font_weight='700', letter_spacing='-1', **{'class': 'heading'}),
        'subtitle': text(36, 121, f'One paper. One colored mark. {len(days)} completed days.', 21, INK, **{'class': 'subtitle'}),
        'description': esc(f'{count_all} paper appearances across {complete} collected days, {days[0]["date"]} to {days[-1]["date"]}. Colors identify primary research categories. {pending_count} labels pending. Missing days are not zero.'),
        'meta': text(960, 37, f'{days[0]["date"]} — {days[-1]["date"]} UTC', 12, text_anchor='end', **{'class': 'mono'}),
        'stats': stats, 'axes': axes, 'annotations': '', 'legend': legend,
        'footer': text(34, 635, 'HUGGING FACE', 12, '#30315d', letter_spacing='1.5') + text(195, 635, 'Daily-list dates · one primary category per paper · ' + ('marks aggregate category counts.' if aggregated else 'one falling mark = one paper.'), 11),
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

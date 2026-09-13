# Running the observatory

Python 3.11+ and `pip install -r requirements.txt` are sufficient. All data lives in this repository. Nothing runs when somebody views the README.

```bash
python3 -m observatory update     # fetch missing data, classify new papers, patch charts
python3 -m observatory ingest     # HF metadata only; no inference
python3 -m observatory classify  # local rules for new papers; preserve historical labels
python3 -m observatory compare-labels  # side-by-side rule predictions and comparison report
python3 -m observatory classify --backend hf  # optional, explicit HF inference
python3 -m observatory render    # entirely offline; patch existing SVGs and tables
python3 -m unittest discover -s tests -v
```

The CLI uses the current UTC date. `--today 2026-09-13` makes the displayed date window reproducible. `--max-batches 1` limits a classifier pilot. `ingest --refresh` explicitly refetches the displayed paper window and model snapshot; ordinary runs reuse completed work. `ingest --include-today` also fetches today's partial list without changing the chart window. A partial date is fetched again after it becomes a completed day. Avoid refresh for styling changes.

## Credentials and scheduled updates

Set `HF_TOKEN` or the existing `HUGGINGFACE_API_KEY` in the environment for authenticated HF metadata access. Local collection also reads these names from the ignored `.env` file, without executing its contents. Default classification uses local keyword rules and needs no token or credits. Only the optional `--backend hf` classifier needs Inference Providers permission and sufficient credits. Metadata access and inference billing are separate.

Testing the replacement token on 2026-09-13 returned **HTTP 402**. All 146 historical papers were subsequently classified in the Codex session, with explicit bootstrap provenance and 30 uncertainty flags. See the [bootstrap audit](reports/labeling-bootstrap.md) and [historical HF probe](data/classifier_probe.json). Automated updates now use local rules, so that inference failure no longer blocks new labels. The HF implementation remains available but is unused by default. Do not paste tokens into chat, commit them, or put them in the configuration file.

For GitHub Actions, configure repository secret `HF_TOKEN` (or retain `HUGGINGFACE_API_KEY`). The workflow tests on code pushes/PRs and refreshes daily at 08:23 UTC or on manual dispatch. After these changes are pushed to `main`, enable Actions and use **Refresh AI observatory → Run workflow** for an initial check. The publication job needs repository contents write permission. The workflow also supports manual dispatch to verify collection and publication before the next scheduled run.

GitHub can delay scheduled runs; images show observation dates. Check the workflow summary and `data/status.json` for deferred work. If another commit lands while the job runs, the normal push fails safely rather than overwriting it; rerun the job against current `main`. Partial collections publish valid data with visible gaps/pending labels, and errors appear in the job summary.

## What changes each day

- Models: six bounded queries cover download/trending lists for text generation, image-to-text generation, and omni (`any-to-any`) models. The display holds up to 48 eligible models, reserving a quarter of positions for trending discoveries. Testing repositories, known quantized conversions, and adapters are excluded. Unknown or nonpositive size/download values are listed in the exclusion table. Candidate totals and exclusions are retained.
- Colors: a small publisher mapping in `config/observatory.json` identifies Qwen/Alibaba, Meta, Google, DeepSeek, Mistral, and OpenAI. Everything else is gray. The mapping refers to the publishing account, not inferred base-model ancestry.
- Size: structured `safetensors.total` from HF, including all experts for MoE models. It can differ from the rounded size in the model’s name. No model weights or configuration files are downloaded to calculate it.
- Papers: yesterday through seven days ago, oldest to newest. Daily-list date determines the column. Same-day duplicates and arXiv versions collapse; a paper listed on separate dates appears in each corresponding day. A completed empty response is zero; missing/incomplete fetches never become zero.
- Labels: local weighted keywords and regexes read full abstracts and titles. Title and contribution-sentence matches receive extra weight; repeated words do not accumulate unlimited votes. New labels include matched phrases, scores, a rationale, and an uncertainty flag. Existing bootstrap/HF labels are preserved. The optional HF backend still uses the configurable `classifier_model`, currently `Qwen/Qwen3-4B-Instruct-2507:nscale`; it is never an automatic fallback.

HF reference pages: [Hub API](https://huggingface.co/docs/hub/api), [download counting](https://huggingface.co/docs/hub/models-download-stats), [inference router](https://huggingface.co/docs/inference-providers/en/index), [rate limits](https://huggingface.co/docs/hub/rate-limits).

## Keyword rules and side-by-side comparison

Edit `config/paper_rules.json` to maintain the vocabulary. Each rule contains a readable name, signed weight, regex, and optional `title`/`abstract` scope. General rules inspect both fields. Text normalization handles case, whitespace, and Unicode hyphens. Strong benchmark/measurement phrases favor Evaluation; learned transition dynamics favor World models, while static-scene terms reduce that score. Similar positive and negative evidence separates robot control, agents, multimodal work, efficient systems, and learning methods.

Every feature votes at most once per paper. The highest category score wins deterministically; weak evidence falls back to Other, and a small winning margin or absent abstract sets an uncertainty flag. Scores are not probabilities. There are no paper-ID exceptions, reference-label lookups, remote calls, or paid fallback in the rule engine.

`compare-labels` computes predictions for every cached paper in `data/labels_rules.jsonl`, compares them with active non-rule references, and writes [label-comparison.md](reports/label-comparison.md) plus machine-readable metrics. It does not alter active labels or chart colors. Current agreement is **134/146 (91.8%)**, or **110/116 (94.8%)** against unflagged bootstrap labels. There are 12 disagreements, 16 rule uncertainty flags, and no low-evidence fallbacks. These are in-sample comparisons with imperfect assistant labels, not measured accuracy on unseen papers.

`classify` and `update` use `classification_backend: rules` by default. They fill missing active labels and refresh previously rule-generated labels when rules change, while preserving bootstrap/HF labels. Cache entries include the input hash, taxonomy version, and hash of the complete rule configuration. Editing the rule file therefore invalidates its cached predictions. If changing engine semantics, also bump the rulebook version. `--backend hf` explicitly selects the retained HF classifier for uncached papers; cached labels remain protected.

## Persistent SVGs

`assets/models.svg` and `assets/papers.svg` are persistent documents. The template is created on first use only. Stable record markers identify each model, each paper appearance, and each day’s labels. An update:

1. Reuses the existing SVG text and static artwork.
2. Adds new keyed marks, patches changed marks, and removes marks outside the selected model sample or displayed paper window.
3. Patches named metadata/axis/legend slots when necessary.
4. Validates the SVG and atomically writes it only if its bytes changed.

Updating downloads must move existing points. Rolling the date window must reposition existing bars. Expanding an axis must adjust coordinates. Those are targeted edits, not a fresh chart build. Domains only expand automatically to minimize unrelated movement. `data/chart_state.json` records them; adjusting its bounds is an explicit rescale operation.

Unchanged records and artwork are retained byte-for-byte. Editing artwork outside the managed slots is supported. Do not remove the slot/record markers. A no-op render preserves file timestamps. Static SVG companions remove the animation through a CSS override. Animations play once, have a completed base state, and respect reduced motion. SVGs contain no JavaScript, external fonts, or network resources.

The animated and static assets were checked locally with Chromium. GitHub’s actual README image proxy and browser combinations still require a check after publication; no remote preview was published. [SVG image restrictions](https://developer.mozilla.org/en-US/docs/Web/SVG/Guides/SVG_as_an_image).

## State and retention

| File | Purpose |
| --- | --- |
| `data/models.json` | Latest selected model snapshot and exclusions |
| `data/model_history/YYYY-MM-DD.jsonl` | Compact model observations per UTC date |
| `data/papers.jsonl` | Canonical paper metadata, abstract, content hash |
| `data/days.json` | Complete daily-list membership and checkpoints |
| `data/labels.jsonl` | Classifications, model/prompt provenance, uncertainty |
| `data/labels_rules.jsonl` | Independent rule predictions, evidence, scores, rule hash |
| `data/label_comparison.json` | Agreement, confusion matrix, disagreements, corpus/rule hashes |
| `data/chart_state.json` | Stable chart scale bounds |
| `data/status.json` | Most recent active collection/classification run and deferrals |

`data/labels.jsonl` contains the 146 initial assistant bootstrap labels and will accumulate new rule labels. Each label records its source (`assistant_bootstrap`, `keyword_rules`, or `hf_inference`), evidence scope, rationale, and uncertainty. Labels are reused by paper ID, content hash, taxonomy version, and prompt version. Bootstrap records use taxonomy/cache version 1 and explicitly identify `codex-session` instead of claiming an HF model produced them. Switching models does not automatically spend credits reclassifying history. Bump the prompt/taxonomy version only when intentionally migrating labels. Invalid HF outputs are repaired once, and only missing/invalid items are resent.

Data accumulates without automatic deletion. Abstracts are kept so label repair works offline up to the inference call. Change `window_days` from `7` to `30` after a month to widen the paper chart; previously collected records and labels are reused. The renderer aggregates category segments instead of animating individual marks when more than 450 appearances are displayed, and says so in the footer. Model history is retained for later analysis but is not animated as invented historical trajectories.

There is no database, external bucket, or authoritative Actions cache. The old script, prompt file, notebook, README, and disabled workflow are under `archive/legacy/`. The old raster pictures were removed during the final repository cleanup.

## Budgets and recovery

The default metadata ceiling is 30 actual HTTP attempts per run, with at most four pages per paper date. Responses are paced; 429/5xx/timeouts have bounded retries. Long retry waits defer work. Authentication, credit, and permission failures stop the affected stage without repeated calls. No full Hub crawl occurs.

Default classification uses zero HTTP attempts. The optional HF backend allows at most 40 HTTP attempts and 180,000 input characters per invocation, with 1,600 output tokens per call. These are workload bounds, not a guaranteed monetary price; provider pricing and tokenization vary. Actual reported token usage is recorded. Metadata may still be fetched for a new date, independently of whether there are new papers.

Collection checkpoints each complete date and labeling checkpoints each successful batch. A page-cap failure retains the previous complete date rather than publishing a truncated count. Rerun after restoring access to finish missing work. The SVG files are individually validated before replacement; a process failure between the two chart writes can leave cards at different snapshots, each with its own dates. Rerun offline rendering to reconcile them.

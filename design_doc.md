# A living AI observatory for the GitHub profile

Status: implementation design, updated 2026-09-13 following the selected direction. Operational instructions are in [OPERATIONS.md](OPERATIONS.md).

## Product and accepted choices

Build a compact, visually distinctive GitHub profile observatory for builders and researchers. It consists of two dark SVG cards, linked to readable tables. GitHub Actions collects HF data daily; viewing the profile makes no API calls. Animations play once and settle into a fully readable chart.

The selected model view is **Option A: adoption versus total parameters**. The paper view covers **seven completed UTC dates**, from seven days ago through yesterday. HF daily-list dates determine paper columns, independently of original publication dates. Sources are HF only. All observations, labels, and chart assets live in this repository and accumulate for a future 30-day display.

The old pipeline is archived under `archive/legacy/`; its workflow is outside `.github/workflows` and cannot run. Previous inference logs are not investigated.

## Model landscape

- Log x: total parameters. Log y: HF downloads over the preceding 30 days. Each dot is one model repository.
- Colors identify publishing accounts: Qwen/Alibaba, Meta, Google, DeepSeek, Mistral, OpenAI, and gray for others. Do not infer a fine-tune's publisher from its base-model family.
- Scope: text-generating LLMs, including VLM and omni task categories. Exclude known adapters, quantized conversions, and synthetic testing repositories. Unknown sizes are disclosed in the companion table, never guessed from a model name.
- Use structured total parameter counts, including all experts in MoE models. Total parameters are not a device-memory estimate.
- A bounded candidate sample combines download and trending lists. Display up to 48 models, reserving a quarter of places for eligible trending discoveries. It is not a census or a capability ranking.
- Stable axes, a subtle dot entrance, restrained publisher colors, and selected direct labels make the chart readable without hover. The full table provides exact counts and source links.

Downloads are an activity proxy, not unique users or benchmark quality; library-specific counting rules affect comparisons. [HF download methodology](https://huggingface.co/docs/hub/models-download-stats).

Potential future views remain adoption-versus-momentum, attention-versus-adoption, a parameter-budget shortlist, and family trajectories. Momentum requires real stored observations; it must not be fabricated from today's counts.

## Research rain

Seven dated columns form a stacked bar chart. Each appearance is a small colored horizontal mark. Marks fall into their category's reserved positions, building bars from the bottom, then hold. A shared count axis preserves comparison. Once a display exceeds 450 appearances, the renderer animates category segments and explicitly discloses that marks are aggregated.

A canonical paper ID appears once per daily list. arXiv versions collapse; an appearance on another day remains a separate appearance. The weekly unique-paper count is distinct from appearances. A complete empty API response is zero. Failed or incomplete pagination produces missing data, never a zero count.

Each paper receives one primary category, based on its central contribution:

| Category | Scope |
| --- | --- |
| Agents | Planning, tools, workflows, autonomous tasks |
| Robotics | Embodied control, navigation, manipulation |
| World models | Learned environment dynamics for prediction or planning |
| Reasoning & learning | Reasoning, training, alignment, general learning |
| Multimodal & generation | Cross-modal understanding and media generation |
| Systems & efficiency | Inference, serving, compression, computation |
| Evaluation & safety | Benchmarks, robustness, interpretability, safety |
| Other / unclear | Outside the taxonomy or insufficient evidence |

A separate neutral pending state represents missing classifications. It is not an LLM-assigned category. The same palette/order is used across dates. This chart describes HF's selected daily papers, not the entire research field.

## Incremental SVG contract

The script **does not regenerate an existing SVG from a template**. A template creates each document once. Persistent record markers identify model dots, paper appearances, and day labels; named slots identify metadata, axes, annotations, and legends.

Updates append new records, patch changed records, and remove expired records. Unchanged records and static artwork remain byte-for-byte. Existing dots move when download counts change. Existing paper groups move when the rolling date window advances. Axis expansion patches affected coordinates. These necessary edits do not rebuild the document scaffold.

The file is validated and atomically replaced only when its bytes change. Atomic disk replacement is an implementation detail, separate from regenerating chart content. An unchanged offline render leaves both bytes and file timestamps unchanged. All rendering is local and makes zero network or inference calls.

Use self-contained CSS animation, no JavaScript or remote fonts, and static companion images. The completed geometry is the base state; reduced-motion styling disables animation. SVGs have titles/descriptions and the README has descriptive alt text plus linked tables. Keep each variant under 200 KB. [SVG image restrictions](https://developer.mozilla.org/en-US/docs/Web/SVG/Guides/SVG_as_an_image).

## Ingestion and state

```text
HF model lists  → bounded collection → dated model observations ──┐
HF daily lists  → canonical papers   → cached local rule labels ──┤
                                                               ↓
                                     existing SVG record/slot patches
                                                               ↓
                                           profile cards + linked tables
```

Persist compact JSON/JSONL in `data/`: model snapshots/history, canonical paper metadata and abstracts, daily membership, cached labels, stable scale state, and operational status. Keep history locally rather than deleting it automatically or depending on an ephemeral Actions cache.

The displayed date range is configurable from 1 to 30 days. Increasing it later reuses accumulated data and labels; only absent dates need backfill. Metadata collection and label classification are independently resumable.

Normal model collection uses six bounded queries across text-generation, image-text-to-text, and any-to-any tasks. No full-Hub traversal or model-file downloads. Model snapshots already collected for the current UTC date are reused. Paper collection fetches missing displayed dates and rechecks yesterday's list when necessary. Older corrections can be requested explicitly with refresh.

Enforce a shared metadata ceiling of 30 actual HTTP attempts, including retries, and a four-page cap per paper date. Respect pagination links and detect repeated pages. Pace requests, retry transient errors with bounded backoff, honor Retry-After, and defer long waits. Preserve complete checkpoints when a source fails. Hub metadata quotas and inference budgets are independent. [HF API](https://huggingface.co/docs/hub/api), [HF rate limits](https://huggingface.co/docs/hub/rate-limits).

## Classification

The default backend is now a deterministic local keyword/regex classifier. Weighted rules in `config/paper_rules.json` inspect full abstracts and titles, with stronger votes for titles and contribution statements. Each feature votes once, avoiding repeated-keyword domination. Positive and negative signals distinguish central contributions from contextual mentions. Low evidence falls back to Other; narrow margins or missing abstracts trigger uncertainty flags.

Rule predictions record all matched phrases, field locations, category scores, margins, and a rule-configuration hash. No reference labels or paper IDs are used by prediction. Changes to the rulebook invalidate the rule cache. The initial 146 assistant classifications remain untouched; new papers get rule labels, and existing rule-generated labels can update when rules change.

The `compare-labels` command runs rules alongside the reference labels and writes separate JSONL predictions, machine-readable metrics, and a report with a confusion matrix and every disagreement. Current agreement is 134/146 (91.8%), and 110/116 (94.8%) on unflagged reference labels. Sixteen rule assignments are flagged uncertain. This is an in-sample diagnostic against imperfect assistant labels, not ground-truth or held-out accuracy. [Comparison report](reports/label-comparison.md).

The original HF classification path is retained and available explicitly through `--backend hf`. It is unused by default and never acts as an automatic fallback. It batches titles/abstracts, validates categories and IDs, repairs missing/invalid outputs once, and checkpoints successful labels. The configured model remains `Qwen/Qwen3-4B-Instruct-2507:nscale`. Only that optional path requires inference permissions and credits; routine rule classification is fully offline.

All backends record their source and evidence. Cached bootstrap/HF records are reused without relabeling or changing provenance. Source metadata distinguishes `assistant_bootstrap`, `keyword_rules`, and `hf_inference`.

## Automation and recovery

The workflow tests code on pushes/PRs and refreshes on a daily schedule or manual dispatch. Read-only jobs run without production inference secrets. The refresh job receives the HF token and contents-write permission, then commits only changed data/assets/reports. It never force-pushes. A concurrent remote change causes a safe push failure and can be retried.

Source errors retain previous valid observations; missing dates/pending labels remain visible. Each chart has its own data dates. A renderer error preserves that chart's last valid file. The two SVG writes are individually atomic, not a multi-file transaction; offline rendering reconciles a partially interrupted update. Git publication commits the resulting state together.

GitHub schedules can be delayed or disabled after inactivity; timestamps and manual dispatch provide recovery. [GitHub scheduled workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

## Validation and remaining external dependency

Tests cover date windows, duplicate/version handling, pagination, missing data, classifier repair/cache behavior, request ceilings, SVG record preservation, offline idempotence, and window rollover. Visual checks use real collected model/paper data and a separate synthetic category fixture for the full color palette, with static and animated browser rendering.

The initial seven-day HF paper collection and model collection succeeded. **Testing a replacement HF token on 2026-09-13 still returned HTTP 402.** At the user's request, all 146 historical papers were classified in the Codex session from title/abstract evidence, with explicit bootstrap provenance and 30 uncertainty flags. These are actual reviewed category assignments, not synthetic fixtures or claimed HF outputs. See the [bootstrap audit](reports/labeling-bootstrap.md). Today and yesterday were empty; September 11 was the latest non-empty daily list and all 26 of its papers are labeled.

Inference access is no longer required for automatic paper labeling. If the optional HF backend is re-enabled later, test a small batch of new papers and review its category boundaries. Existing bootstrap labels remain cached. After publication, verify GitHub's actual README image embedding/cache behavior; local Chromium checks do not establish all GitHub/browser combinations. No remote publishing was performed during implementation.

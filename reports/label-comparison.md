# Keyword-rule label comparison

Agreement: **134/146 (91.8%)**. On references without an uncertainty flag: **110/116 (94.8%)**.

This is agreement with the existing non-rule labels (initially Codex bootstrap), not ground-truth accuracy. Rules were developed with this same historical corpus available, so this is an in-sample diagnostic, not a held-out estimate of future accuracy. No reference labels or paper IDs are used as inputs to prediction. Reference labels are not overwritten.

16 rule assignments are flagged uncertain; 0 use the low-evidence fallback. Scores and margins are heuristic votes, not calibrated probabilities.

## Per-category agreement

| Category | Reference count | Rules count | Exact matches | Recall vs reference |
| --- | ---: | ---: | ---: | ---: |
| Agents | 17 | 19 | 14 | 82.4% |
| Robotics | 7 | 6 | 6 | 85.7% |
| World models | 6 | 6 | 6 | 100.0% |
| Reasoning & learning | 20 | 24 | 20 | 100.0% |
| Multimodal & generation | 36 | 35 | 33 | 91.7% |
| Systems & efficiency | 18 | 16 | 16 | 88.9% |
| Evaluation & safety | 41 | 39 | 38 | 92.7% |
| Other / unclear | 1 | 1 | 1 | 100.0% |

## Confusion matrix

Rows = reference; columns = rules.

| Reference / rules | agents | robotics | world_models | reasoning | multimodal | systems | evaluation | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agents | 14 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| robotics | 1 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| world_models | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| reasoning | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 |
| multimodal | 1 | 0 | 0 | 2 | 33 | 0 | 0 | 0 |
| systems | 0 | 0 | 0 | 1 | 1 | 16 | 0 | 0 |
| evaluation | 3 | 0 | 0 | 0 | 0 | 0 | 38 | 0 |
| other | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## Disagreements

| Paper | Reference | Rules | Reference uncertain? | Rules uncertain? | Evidence |
| --- | --- | --- | --- | --- | --- |
| [Graph Machine: Towards Better Pretraining via Edges](https://huggingface.co/papers/2609.02881) | systems | reasoning | True | True | Matched learning mechanics; score 4, margin 0. |
| [EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness?](https://huggingface.co/papers/2609.04280) | evaluation | agents | False | True | Matched agent system, self improvement, agent contribution; score 17.4, margin 0.6. |
| [Ask Before You Optimize: Dynamic Pre-Formulation Clarification for Interactive Optimization](https://huggingface.co/papers/2609.05258) | agents | evaluation | True | False | Matched benchmark contribution, evaluation purpose; score 16.8, margin 7.8. |
| [EvoSafeHarness: Evolving Model- and Domain-Specific Harnesses for Securing Agents](https://huggingface.co/papers/2609.05903) | evaluation | agents | False | False | Matched agent system, agent contribution; score 10.2, margin 6.2. |
| [Counter-Swarm Doctrine: Containing Coordinated Agent Intrusions](https://huggingface.co/papers/2609.06140) | evaluation | agents | False | False | Matched agent system; score 6, margin 6. |
| [OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution](https://huggingface.co/papers/2609.06490) | multimodal | reasoning | False | False | Matched training methods, posttraining contribution, learning mechanics; score 16.2, margin 6.2. |
| [Omni Interaction Agent Technical Report](https://huggingface.co/papers/2609.08977) | multimodal | agents | True | True | Matched agent system, agent contribution; score 10.2, margin 2. |
| [AgenticGen: Reward-Guided Agentic Video Generation for Advertising](https://huggingface.co/papers/2609.09187) | agents | multimodal | True | True | Matched visual generation, multimodal models; score 13, margin 2.8. |
| [Show-Harness: Just a VLM Agent Can Play Robots](https://huggingface.co/papers/2609.10522) | robotics | agents | True | True | Matched agent system, agent contribution; score 10.2, margin 2. |
| [DRG-MAPPO: Hierarchical Dynamic Role-Graph Multi-Agent Reinforcement Learning for Cooperative Air Combat](https://huggingface.co/papers/2609.11155) | agents | reasoning | True | False | Matched training methods, posttraining contribution, learning mechanics; score 13.8, margin 3.6. |
| [Generative Late-Interaction Embeddings For Visual Document Retrieval](https://huggingface.co/papers/2609.11808) | systems | multimodal | False | True | Matched visual perception; score 8, margin 1. |
| [SenseNova-U1.5: Towards Native Unified Visual Intelligence](https://huggingface.co/papers/2609.11929) | multimodal | reasoning | False | False | Matched training methods, learning mechanics; score 6, margin 3. |

Rules: [paper_rules.json](../config/paper_rules.json). Full matched phrases, scores, and provenance: [labels_rules.jsonl](../data/labels_rules.jsonl). Machine-readable metrics: [label_comparison.json](../data/label_comparison.json).

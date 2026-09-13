# Historical paper labeling audit — 2026-09-13

All 146 collected papers have a primary category. These labels were assigned in the Codex assistant session from titles and abstract excerpts; full abstracts were inspected for selected ambiguous cases. No PDFs were read. These are not human expert labels or successful HF inference outputs. Thirty assignments are flagged uncertain; the flag describes category-boundary judgment, not a calibrated probability.

The replacement local token was selected correctly, but the configured HF classifier returned HTTP 402. That probe produced no labels and used no reported inference tokens. Today and yesterday had no daily-list papers. All 26 papers on the latest non-empty day (September 11) are included in this bootstrap.

## Coverage

| HF list date | Papers | Labeled | Status |
| --- | ---: | ---: | --- |
| 2026-09-06 | 0 | 0 | completed |
| 2026-09-07 | 28 | 28 | completed |
| 2026-09-08 | 12 | 12 | completed |
| 2026-09-09 | 48 | 48 | completed |
| 2026-09-10 | 32 | 32 | completed |
| 2026-09-11 | 26 | 26 | completed |
| 2026-09-12 | 0 | 0 | completed |
| 2026-09-13 | 0 | 0 | partial |

## Category totals

| Category | Papers |
| --- | ---: |
| Agents | 17 |
| Robotics | 7 |
| World models | 6 |
| Reasoning & learning | 20 |
| Multimodal & generation | 36 |
| Systems & efficiency | 18 |
| Evaluation & safety | 41 |
| Other / unclear | 1 |

## Decisions worth reviewing

Static 3D scene generation is Multimodal, even when the title says “world.” Learned action-conditioned simulation is World models. Programmable World Model is flagged as a hybrid boundary case because explicit transition rules drive a learned visual renderer. Benchmark papers are Evaluation even when they test robots or agents.

| Paper | Assigned category | Reason |
| --- | --- | --- |
| [The Price of Sparsity: Sufficient Conditions for Sparse Recovery using Sparse and Sparsified Measurements](https://huggingface.co/papers/2509.01809) | Reasoning & learning | Statistical sample-complexity guarantees for sparse recovery; adjacent to general learning theory. |
| [From Reweighting to Rewriting: Unlocking the Intervention Effects of Influential Samples in Training Data Attribution](https://huggingface.co/papers/2609.02771) | Reasoning & learning | Influence-guided rewriting of training responses changes model behavior; also evaluates attribution methods. |
| [Graph Machine: Towards Better Pretraining via Edges](https://huggingface.co/papers/2609.02881) | Systems & efficiency | Sparse dynamic routing architecture reduces sequence-processing complexity while retaining pretraining quality. |
| [The Attention Triangle in Audio-Video Models](https://huggingface.co/papers/2609.03586) | Evaluation & safety | Mechanistically probes cross-modal attention routing and semantic leakage in generative models. |
| [AdaptVPR: Route-Aware Hard Positive Generation for Robust Visual Place Recognition](https://huggingface.co/papers/2609.04369) | Multimodal & generation | Generative visual augmentation improves place recognition; localization is the downstream application. |
| [MaxKernel: Agentic Kernel Generation for TPUs](https://huggingface.co/papers/2609.04523) | Agents | Multi-agent planning, coding, debugging, and profiling automate TPU kernel development. |
| [Ask Before You Optimize: Dynamic Pre-Formulation Clarification for Interactive Optimization](https://huggingface.co/papers/2609.05258) | Agents | An interactive clarification policy resolves missing requirements before optimization-model formulation. |
| [SceneMosaic: Efficient and Diverse Simulation-Ready Scene Generation via Hybrid Agentic Layout Evolution](https://huggingface.co/papers/2609.05594) | Multimodal & generation | Generates simulation-ready static scene layouts; agentic search is the generation mechanism. |
| [VidaForge: Open Research Infrastructure for Video Pretraining Data Recipes](https://huggingface.co/papers/2609.06652) | Systems & efficiency | Reusable video-data infrastructure supports reproducible pretraining curation and recipe experiments. |
| [PARSER: Read in Parallel, Reason in Depth for Long-Context LLM Agents](https://huggingface.co/papers/2609.06702) | Agents | Parallel readers decouple long-document traversal from an agent's subsequent reasoning. |
| [DianShi-RxnDB: A Large-Scale, Fine-Grained Organic Reaction Data Platform Built via a Fully Automated Pipeline for Researchers and AI Agents](https://huggingface.co/papers/2609.06703) | Other / unclear | A domain-specific organic-reaction data platform; its main contribution is structured chemistry data. |
| [CARDEA: Auditable Reasoning Grounded in Spatial Evidence for End-to-End Coronary Angiography Interpretation](https://huggingface.co/papers/2609.06931) | Multimodal & generation | Grounds medical video interpretation and reporting in spatial visual evidence. |
| [Measuring Language Transfer in Robot Policies: Adding Greek to a Cosmos3 Vision-Language-Action Policy](https://huggingface.co/papers/2609.07470) | Evaluation & safety | Evaluates and validates measurement of low-resource language transfer in robot policies. |
| [Kalman Delta Networks: Uncertainty-aware Associative Memory](https://huggingface.co/papers/2609.07816) | Reasoning & learning | Uncertainty-aware associative-memory updates improve the learning behavior of linear-attention models. |
| [A*-Thought-V2: Efficient Latent Reasoning via Geometric Dynamics of LLM](https://huggingface.co/papers/2609.07821) | Reasoning & learning | Geometric latent reasoning compresses chains of thought while retaining reasoning information. |
| [SynthGait-19K: A Physically Grounded Synthetic Video Dataset for Gait Parameter Estimation](https://huggingface.co/papers/2609.08108) | Multimodal & generation | Physically grounded synthetic videos provide training supervision for visual gait estimation. |
| [NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness](https://huggingface.co/papers/2609.08183) | Agents | An agent routing harness gathers feedback to drive recursive agent-model post-training. |
| [Environments as Scaffold: Enriching Feedback to Bootstrap Self-Evolving Agents in Long-Horizon Tasks](https://huggingface.co/papers/2609.08404) | Agents | Environment-side feedback scaffolds train agents for long-horizon interaction. |
| [PlannerForge: LLM Agents for Scenario-Based Testing of Motion Planners in Autonomous Driving](https://huggingface.co/papers/2609.08965) | Agents | LLM orchestration unifies scenario-generation, testing, and improvement tools for driving planners. |
| [Omni Interaction Agent Technical Report](https://huggingface.co/papers/2609.08977) | Multimodal & generation | A unified model integrates streaming omni perception and full-duplex interaction with agent capabilities. |
| [ActReview: Rebuttal-Guided Training Data and Rubric Rewards for Actionable Peer Review Generation](https://huggingface.co/papers/2609.09076) | Reasoning & learning | Rebuttal-derived supervision and rubric rewards post-train models for actionable peer review. |
| [NOAH: Learning the Full Patient Journey. A Longitudinal Multimodal Time-Aware Model for Representation and Forecasting](https://huggingface.co/papers/2609.09140) | Multimodal & generation | Fuses longitudinal clinical modalities for patient representation and forecasting. |
| [Studying Image Tokenizers as Visual Languages in Unified Multimodal Models](https://huggingface.co/papers/2609.09143) | Evaluation & safety | A controlled testbed analyzes image-tokenizer behavior in joint multimodal training. |
| [AgenticGen: Reward-Guided Agentic Video Generation for Advertising](https://huggingface.co/papers/2609.09187) | Agents | Reward-guided planning and tool policies orchestrate product-conditioned advertising video creation. |
| [Show-Harness: Just a VLM Agent Can Play Robots](https://huggingface.co/papers/2609.10522) | Robotics | A semantic action interface translates VLM intent into robot control across embodiments. |
| [Programmable World Model](https://huggingface.co/papers/2609.10540) | World models | A hybrid world simulator couples explicit programmed state transitions with learned video rendering. |
| [Beyond Solver Verdicts: Generative Reward Models for Autoformalization](https://huggingface.co/papers/2609.11085) | Evaluation & safety | A generative verifier detects unfaithful formal translations despite successful solver verdicts. |
| [DRG-MAPPO: Hierarchical Dynamic Role-Graph Multi-Agent Reinforcement Learning for Cooperative Air Combat](https://huggingface.co/papers/2609.11155) | Agents | Hierarchical role graphs coordinate multi-agent reinforcement-learning policies in a simulated control setting. |
| [World in World: Explore the World with World Models](https://huggingface.co/papers/2609.11548) | World models | A control interface supports consistent long-horizon exploration with a causal video world model. |
| [Memory as Plans: World-Action Modeling with Memory-Grounded Planning](https://huggingface.co/papers/2609.11561) | Robotics | Memory-grounded plans and progress-conditioned execution improve long-horizon robot manipulation. |

Full provenance and content hashes are in [labels.jsonl](../data/labels.jsonl). These cached labels prevent future HF calls for unchanged historical papers.

# Multi-Corpus Research Benchmark Report (2026-11-24)

> **Research-Only Notice**: All models and benchmarks below are strictly offline research experiments across public English multimodal corpora and anonymized local L0 kinematics windows (zero PII, excluded from `report_frontend/` scoring).

## 0. Multimodal Feature Extraction Throughput

- **Total Clips Extracted**: 598 / 598 (failed: 0)
- **Wall Time & Throughput**: 365.64s (1.635 clips/sec across 14 CPU workers, 8 frames/clip)

| Corpus | Clips | Features | Mean Face Detect | Silero ONNX Clips |
|---|---:|---:|---:|---:|
| mit | 138 | 167 | 0.989 | 138/138 |
| recruitview | 160 | 167 | 1.000 | 160/160 |
| chalearn | 300 | 167 | 0.994 | 300/300 |

## 1. Experiment 1 · MIT Interview (`N=138`, 69 Participants, 5-Fold GroupKFold)

- **GPU Hardware Benchmark**: `Tesla P40` (`torch 2.5.1+cu121`, strict `torch.float32`, peak VRAM `17.81 MB`, 5-fold total `4.6054s`)
- **Best Model (Macro-18 Pearson r)**: `Behavior-RF`

| Model | Macro-18 $r$ | Macro-18 $\rho$ | Core-8 $r$ | Macro-18 RMSE |
|---|---:|---:|---:|---:|
| Behavior-Ridge | 0.3898 | 0.3733 | 0.3741 | 0.6208 |
| Behavior-RF | 0.4023 | 0.3976 | 0.3726 | 0.6231 |
| Behavior-HistGBT | 0.3939 | 0.3825 | 0.3443 | 0.6234 |
| Text-TFIDF-Ridge | 0.2100 | 0.1974 | 0.2309 | 0.6809 |
| Early-Fusion-Ridge | 0.3919 | 0.3767 | 0.3747 | 0.6213 |
| Late-Fusion | 0.3276 | 0.3274 | 0.2687 | 0.6198 |
| Extracted-AV-Ridge | 0.3797 | 0.3789 | 0.3395 | 0.6289 |
| GPU-FP32-MLP | 0.3627 | 0.3473 | 0.3610 | 0.6783 |

### Per-Target Breakdown for `Behavior-RF` (Core-8 Highlights)

| Target | Pearson $r$ | Spearman $\rho$ | RMSE |
|---|---:|---:|---:|
| Overall | 0.4738 | 0.4360 | 0.5691 |
| RecommendHiring | 0.5090 | 0.4757 | 0.6342 |
| EyeContact | 0.0759 | 0.1138 | 0.7760 |
| Smiled | 0.6701 | 0.6642 | 0.7823 |
| SpeakingRate | 0.2019 | 0.1840 | 0.3648 |
| Paused | 0.3612 | 0.3558 | 0.4571 |
| StructuredAnswers | 0.4110 | 0.4296 | 0.6372 |
| Focused | 0.2775 | 0.2242 | 0.5083 |

## 2. Experiment 2 · RecruitView (`N=2,011` Text + `N=160` Multimodal, 5-Fold GroupKFold by `user_no`)

### Track A · Full Transcript Corpus (`N=2011`, `331` Unique Users)

| Model | Macro-12 $r$ | Macro-12 $\rho$ | BigFive-6 $r$ | Macro-12 RMSE |
|---|---:|---:|---:|---:|
| Text-TFIDF-Ridge | 0.2171 | 0.2442 | 0.1853 | 1.0633 |
| Text-SVD-HistGBT | 0.2325 | 0.2412 | 0.1989 | 1.0607 |

### Track B · Multimodal Subset (`N=160`, `80` Unique Users, `161` Behavior Features)

| Model | Macro-12 $r$ | BigFive-6 $r$ | 1 - MAE_norm | Macro-12 RMSE |
|---|---:|---:|---:|---:|
| Behavior-Ridge | 0.1421 | 0.1299 | 0.9529 | 0.9678 |
| Behavior-RF | 0.2118 | 0.1789 | 0.9538 | 0.9466 |
| Behavior-HistGBT | 0.2145 | 0.2069 | 0.9504 | 0.9946 |
| Text-TFIDF-Ridge | -0.0261 | -0.0095 | 0.9521 | 0.9847 |
| Early-Fusion-Ridge | 0.1444 | 0.1386 | 0.9530 | 0.9675 |
| Late-Fusion | 0.1035 | 0.0958 | 0.9528 | 0.9641 |

## 3. Experiment 3 · ChaLearn First Impressions V2 (`train=180`, `val=60`, `test=60`)

- **Best Model**: `AudioVisual-Ridge` (`140` visual + `21` acoustic features)

| Model | Official `1 - MAE` | Pearson $r$ | Spearman $\rho$ | RMSE |
|---|---:|---:|---:|---:|
| AudioVisual-Ridge | 0.9126 | 0.4563 | 0.4630 | 0.1105 |
| AudioVisual-RF | 0.9096 | 0.4298 | 0.4174 | 0.1130 |
| AudioVisual-HistGBT | 0.9095 | 0.3498 | 0.3328 | 0.1157 |
| Visual-only-Ridge | 0.9042 | 0.3750 | 0.3893 | 0.1182 |
| Acoustic-only-Ridge | 0.9060 | 0.3322 | 0.3049 | 0.1175 |

### Per-Trait ChaLearn Test Metrics (`AudioVisual-Ridge`)

| Trait | `1 - MAE` | Pearson $r$ | Spearman $\rho$ | RMSE |
|---|---:|---:|---:|---:|
| extraversion | 0.9202 | 0.6338 | 0.6751 | 0.1043 |
| neuroticism | 0.9121 | 0.4827 | 0.4239 | 0.1084 |
| agreeableness | 0.9059 | 0.3510 | 0.4670 | 0.1191 |
| conscientiousness | 0.9072 | 0.3936 | 0.4061 | 0.1140 |
| openness | 0.9112 | 0.4441 | 0.3889 | 0.1140 |
| interview | 0.9190 | 0.4323 | 0.4170 | 0.1030 |

## 4. Experiment 4 · Cross-Corpus Behavioral Regime Transfer (`English <-> 47 Chinese Sessions`)

- **Chinese Reference Corpus**: 47 sessions (5666 windows, 21 with voice)
- **English Transfer Corpus**: 598 clips ({'mit': 138, 'recruitview': 160, 'chalearn': 300})

| Regime Model | Silhouette (Chinese) | Davies-Bouldin | ARI vs Chinese | NMI vs Chinese | Agreement |
|---|---:|---:|---:|---:|---:|
| Chinese-only-GMM (5-fold GroupKFold) | 0.3372 | 1.2687 | 0.9785 | 0.9513 | 0.9892 |
| English-pretrained-GMM (Zero-Shot on Chinese) | 0.2829 | 1.1925 | 0.4433 | 0.5050 | 0.7076 |
| Bilingual-Joint-GMM (5-fold GroupKFold on Chinese) | 0.3562 | 1.4088 | 0.8971 | 0.8315 | 0.9486 |

## 5. Saved Research Model Artifacts (`models/research/`)

| Artifact | Size (KB) |
|---|---:|
| `behavior_regime_multicorpus.joblib` | 4.1 |
| `chalearn_bigfive_behavioral.joblib` | 14.6 |
| `mit_interview_multimodal.joblib` | 279.9 |
| `recruitview_bigfive_multimodal.joblib` | 2146.5 |

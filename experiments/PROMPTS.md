# CueCoT 实验实施提示词（AAAI 2027 质量版）

每个阶段的提示词可独立使用，直接复制给 Claude 即可执行。

---

## 阶段一：批量特征提取 ✅ 已完成

`extract_features.py` 已实现并通过 2 视频测试。2011 个视频全量特征提取正在后台运行。

**如需重跑：**
```
cd D:\jingxin\experiments
C:\Users\Lenovo\.conda\envs\jingxin\python.exe extract_features.py --workers 8
```

---

## 阶段二：时序聚合

**提示词：**

```
## 任务

在 D:\jingxin\experiments\ 下创建 aggregate_features.py，将阶段一产出的逐帧
CSV 文件聚合为固定长度的特征向量。

## 要求

### 输入
- D:\jingxin\experiments\results\features\vid_XXXX\ 下每个视频的 face.csv, gesture.csv, voice.csv, metadata.json

### 输出
对每个视频生成 feature_vector.json：
  D:\jingxin\experiments\results\features\vid_XXXX\feature_vector.json

### 聚合策略（对应论文 Section 4.2）

#### 面部特征（face.csv 的每列）
对每个数值列计算：
  - mean, std, min, max          基础统计
  - slope（线性回归拟合的趋势）    趋势指标
  - volatility（std/mean，mean≈0 时用 std*10 替代）
  - 滑动窗口：视频按时间分 3 段，分别计算每段 mean → start_avg, mid_avg, end_avg

#### 手势特征（gesture.csv 的每列）
同面部：mean, std, slope, volatility

#### 语音特征（voice.csv）
voice.csv 通常只有一行，直接使用所有列作为特征。
如有多行，计算 mean 和 std。

### 特征向量格式
```json
{
  "video_id": "vid_0001",
  "user_no": "147",
  "question_id": "1",
  "duration": "long",
  "face": {
    "au1_inner_brow_raise": {"mean": 0.12, "std": 0.05, "slope": 0.001, "volatility": 0.42, "start_avg": 0.10, "mid_avg": 0.13, "end_avg": 0.11},
    "au12_smile": {...},
    "gaze_deviation": {...}
  },
  "gesture": {
    "left_hand_jitter": {"mean": 0.015, ...},
    "right_arm_angle": {"mean": 85.3, ...}
  },
  "voice": {
    "pitch_mean": 234.5, "pitch_std": 34.2, "energy_mean": 0.023,
    "speech_ratio": 0.78, "pause_duration_mean": 0.45, "pause_frequency": 12.0
  },
  "targets": {
    "openness": -0.65, "conscientiousness": -0.05, "extraversion": -0.69,
    "agreeableness": -0.29, "neuroticism": 0.19, "overall_personality": -0.03,
    "interview_score": -0.92, "answer_score": -0.80, "speaking_skills": -0.77,
    "confidence_score": -0.36, "facial_expression": -0.82, "overall_performance": -0.46
  }
}
```

### 全局汇总
- 生成 all_features.jsonl（每行一个视频的完整特征向量 + targets + user_no + question_id）
- 生成 all_features.npy [n_videos, total_feature_dim] 和 targets.npy [n_videos, 12]
- 生成 feature_names.txt（每行一个特征名，与矩阵列对应）
- 生成 data_summary.json，记录：
  - 总视频数、排除数（低质量：face_detection_rate < 50%）
  - 总特征维度
  - 每维 target 的均值/标准差（确认与 metadata.jsonl 一致）
  - 各模态特征数的分布

### 数据质量检查
- 检查 NaN/Inf 值，标记包含 NaN 的视频，决定是填充（用中位数）还是排除
- 检查 all_features.npy 是否有常数列（std=0），记录这些列名
- 输出 quality_report.json 记录所有质量检查结果

### 验证
- all_features.jsonl 行数 = 成功处理的视频数
- 随机抽取 5 个视频，手动验证 feature_vector.json 中的统计量与原始 CSV 的一致性
```

---

## 阶段三：CoT Prompt 与 LLM 推理

**提示词：**

```
## 任务

在 D:\jingxin\experiments\ 下创建 cot_prompts.py 和 llm_reasoning.py，
实现论文核心的三阶段 CoT 推理模块。

## 背景

这是 CueCoT 论文的核心创新——通过三阶段 CoT（Observation → Inference →
Assessment）引导 LLM 进行结构化面试评估推理。该模块的质量直接决定论文
novelty 的可行性。

## cot_prompts.py 要求

### 功能一：自然语言序列化（Training-free variant）
将 feature_vector.json 转换为专业的行为心理学特征描述文本。

**设计原则**：
- 每个数值配上下文解释（如 "gaze deviation = 0.04 → 视线高度稳定，表明持续注意力"）
- 使用行为心理学专业术语（affect display, regulatory behavior, paralinguistic cues）
- 按论文 Section 4.4 的三阶段结构组织：raw signals → psychological states → assessment

### 功能二：三阶段 CoT Prompt 模板

Stage 1 - Observation Prompt（行为信号观测）:
要求 LLM 从特征中提取可观察的行为事实，禁止做心理推断。输出结构化 JSON。

Stage 2 - Inference Prompt（心理状态推断）:
要求 LLM 基于 Stage 1 的观察，推断心理状态（engagement, confidence, anxiety, fluency）。
每个推断必须引用 Stage 1 的具体观察作为证据。输出结构化 JSON。

Stage 3 - Assessment Prompt（能力评估）:
要求 LLM 基于前两阶段，对 12 个维度打分。每个分数附带：
  1. score（连续值，匹配数据集的归一化尺度，约 [-1, 1] 范围）
  2. evidence_chain（从 Stage 1 观察 → Stage 2 推断 → 此分数的一行追溯链）

### 功能三：Baseline Prompt 模板
- **Direct LLM (no CoT)**：特征描述 + "Rate on 12 dimensions. Output JSON only."
- **Generic CoT**：特征描述 + "Let's think step by step." + 打分要求
- **Ablation: Skip Stage 1**：直接给特征 + "Infer psychological states and assess"
- **Ablation: Skip Stage 2**：Stage 1 观察 + "Directly assess from observations"

### Prompt 质量要求
- 每个 prompt 要求 LLM 以 **严格 JSON 格式** 输出（便于自动解析）
- Prompt 中包含 few-shot 示例（2-3 个完整推理链，人工撰写高质量示例）
- 明确告知 LLM 分数的 scale（"scores are normalized, centered around 0, typically ranging from -2 to 2"）

## llm_reasoning.py 要求

### LLM 调用封装
- 使用 openai Python SDK 调用 GPT-4o（gpt-4o-2024-08-06）
- API key 从环境变量 OPENAI_API_KEY 读取
- 关键函数：
  - `run_cot_full(feature_vector)` → {scores, reasoning_chain, tokens_used}
  - `run_direct(feature_vector)` → {scores, tokens_used}
  - `run_generic_cot(feature_vector)` → {scores, reasoning_chain, tokens_used}
  - `run_cot_skip_stage1(feature_vector)` → {scores, ...}（消融用）
  - `run_cot_skip_stage2(feature_vector)` → {scores, ...}（消融用）

### 输出解析与容错
- 主解析：`json.loads()` 从 LLM 输出提取 JSON 块
- 兜底：正则提取 "dimension_name": float_pattern
- 解析失败时记录 raw_output 并标记为 parse_error，不丢弃数据
- 实现 `validate_scores(scores)` 检查分数是否在合理范围

### 批量运行
- 读取 all_features.jsonl
- 对每个视频依次运行所有 variant（全 CoT / no CoT / generic CoT / skip S1 / skip S2）
- 每个 video × method 组合输出一行到 llm_results.jsonl
- 实现速率限制 + 指数退避重试（tenacity 库）
- 断点续跑：已有结果的行不重复调用

### LLM 鲁棒性测试（AAAI 级别要求）
- 对 test set 中的每个视频，每种方法运行 temperature ∈ {0.0, 0.3, 0.7} 各一次
- 报告不同 temperature 下分数的均值和方差（衡量 LLM 推理稳定性）
- 此为论文补充材料，主实验用 temperature=0.0

### 成本管理
- 先在 20 个视频上测试所有 variant，估算每个的 token 消耗
- 打印预估全量成本，确认后再全跑
- 建议主实验用 GPT-4o，消融中加 GPT-4o-mini 对比
```

---

## 阶段四：Baseline 实现（AAAI 质量版）

**提示词：**

```
## 任务

在 D:\jingxin\experiments\ 下创建 baselines.py，实现论文中所有对比方法。
**必须达到 AAAI 2027 的实验严谨性标准。**

## 全局实验规范

### 随机种子与重复实验（CRITICAL）
- 所有涉及随机性的方法（MLP, Attention-LSTM, 数据划分）必须运行 5 个独立种子
- 种子列表：SEEDS = [42, 123, 456, 789, 1024]
- 报告格式：mean ± std (5 seeds)，如 "0.342 ± 0.018"
- 所有随机模块（numpy, random, torch）在每次运行前重置种子

### 数据划分（防泄露）
- 按 user_no 分层划分 train/val/test = 7:1:2
- 同一 user_no 的所有视频必须在同一 split
- 保存 split_info.json（train_users, val_users, test_users 列表）
- 所有方法使用相同的 split（包括 LLM-based 方法）

### 评估指标
- 主指标：MSE（回归任务的标准指标）
- 辅助指标：Pearson r, Spearman ρ, MAE
- 每个维度单独计算，再计算 12 维平均
- 二类分组：Personality traits (6维) 和 Interview Performance (6维) 分别报告 Macro Avg

### 统计显著性
- 对 CueCoT vs 最佳 baseline 做 paired bootstrap test（10,000 次重采样）
- 报告 95% bootstrap 置信区间
- 报告 p 值（双侧）

## Baseline 实现

### 1. Unimodal Regressors
每模态 × 每维度：
  - **SVR**（RBF kernel, C ∈ {0.1,1,10,100}, gamma ∈ {0.001,0.01,0.1,1}，GridSearchCV）
  - **Ridge Regression**（alpha ∈ {0.01,0.1,1,10,100}，GridSearchCV）
  - **Random Forest**（n_estimators=200, max_depth ∈ {5,10,20,None}）
  超参搜索在 val split 上进行，选最佳配置后在 test 上评估。

### 2. Multimodal Fusion
  - **Early Fusion MLP**：3 层 [256,128,64], dropout=0.2, AdamW, lr=1e-3, 200 epochs, early stopping patience=20
  - **Late Fusion**：三个单模态最佳模型结果取加权平均（权重在 val 上调）
  - **CRMF（RecruitView 原论文 baseline）**：阅读 RecruitView 论文 Section 4 实现 Cross-Modal Regression with Manifold Fusion。最低要求复现其核心思路（manifold alignment + regression），或至少引用其报告数字（Spearman ρ=0.568, C-Index=0.718）

### 3. Attention-LSTM Fusion
参考 malik2022explainable：
  - 输入时序特征 → BiLSTM(128) → Multi-head Self-Attention(4 heads) → FC(64) → 12-dim output
  - PyTorch 实现，AdamW, lr=1e-3, 100 epochs, early stopping
  - 如果逐帧特征难以对齐，降级为：聚合特征 + Self-Attention pooling

### 4. Direct LLM Prompting
调用 llm_reasoning.py 的 run_direct()

### 5. LLM with Generic CoT
调用 llm_reasoning.py 的 run_generic_cot()

### 6. Text-only LLM Baseline（新增）
仅使用 transcript + gemini_summary 文本输入给 GPT-4o，不使用多模态特征。
目的：验证多模态行为信号是否提供了文本之外的增量信息。
这是 AAAI 审稿人极有可能问的问题。

### 超参数管理
- 所有超参数在 baselines.py 顶部集中定义，加注释说明选择理由
- 超参数搜索结果保存为 hyperparam_search.json
- 最终选择的超参数在论文附录中报告

### 输出
- results/baseline_results.json：每个方法 × 每 seed × 每维度的指标
- results/baseline_summary.json：汇总均值 ± std + 95% CI
- results/split_info.json
- results/hyperparam_search.json
```

---

## 阶段五：CueCoT 完整实验 + 消融 + 统计分析

**提示词：**

```
## 任务

在 D:\jingxin\experiments\ 下创建 run_experiments.py 和 ablation.py，
运行完整实验并产出 AAAI 级别的实验报告。

## run_experiments.py 要求

### Learned Projection Variant（论文核心）
- 实现 ModalityProjector：每个模态 → 2 层 MLP [256, 128] → GELU → d_proj
- Training objective: L = L_reg + λ * L_contrast
  - L_reg: MSE(预测分数, ground truth)
  - L_contrast: InfoNCE loss，同一样本三模态投影互为正样本对
  - λ 在 {0.01, 0.05, 0.1, 0.5} 中通过 val 选择
- 用 train split 训练投影器（LLM 冻结），5 个种子
- 推理时：投影特征 + CoT prompt → LLM → 分数

### Training-free Variant
- 自然语言特征描述 + CoT prompt → LLM → 分数
- 无需训练，但也要跑 5 个 LLM temperature 重复（temperature=0 为主）

### 主实验执行流程
1. 在 test split 上评估所有方法（baselines + CueCoT 两个 variant）
2. 每方法 × 每维度计算 4 个指标（MSE, MAE, Pearson r, Spearman ρ）
3. 生成主结果表（LaTeX 格式）

### 统计分析（单独脚本 stats_analysis.py）
- CueCoT vs 每种 baseline 的 paired bootstrap test（10k iterations）
- 报告 95% CI 和 p 值
- 输出完整的统计报告供论文使用

## ablation.py 要求

### 消融一：移除推理阶段
- 无 Stage 1（Observation）：特征 → 直接推断 + 评估
- 无 Stage 2（Inference）：观察 → 直接评估
- 仅 Stage 1+2（无评估）：定性分析推理链质量

### 消融二：移除模态
- 所有 7 种模态组合（3 单模态 + 3 双模态 + 1 三模态）
- 对比 learned projection 和 training-free 在每种组合下的表现

### 消融三：Projection 设计
- Learned MLP (2-layer, GELU)
- Linear projection (1-layer, no activation)
- Training-free text serialization
- 额外对比：是否使用 contrastive loss

### 消融四：LLM 选择
- GPT-4o vs GPT-4o-mini（成本和性能权衡）
- 如果成本允许，加 Gemini 2.5 Flash 或 Claude Haiku 作为跨模型对比

### 消融五：CoT Prompt 设计（新增）
- 有无 few-shot 示例
- 有无 JSON 格式约束
- 有无特征上下文解释（裸数值 vs 带解释的描述）

### 消融六：特征聚合粒度（新增）
- 帧级特征 vs 窗口级（3s 窗口）vs 视频级全局聚合
- 验证论文中选择的全局聚合 + 窗口统计是否最优

## 错误分析（单独脚本 error_analysis.py）

按以下维度拆解 CueCoT 和最佳 baseline 的表现：
  - 视频时长（short/medium/long）
  - 问题类型（按 question_id 分组）
  - 预测误差分布（残差 histogram，检查 bias）
  - 高误差案例（top-20 残差案例，检查共性和 failure mode）
  - 输出 error_analysis.json + 可视化图表

## 输出汇总
- results/main_results.json（每方法每维度 4 指标 × 5 seeds）
- results/main_results_summary.json（mean ± std + CI + p 值）
- results/ablation_results.json
- results/stats_analysis.json
- results/error_analysis.json
- results/main_table.tex（可直接 \input 到论文中）
```

---

## 阶段六：论文图表、定性分析、人类评估、可复现性

**提示词：**

```
## 任务

基于阶段四、五的实验结果，生成 AAAI 2027 论文所需的全部材料。

## 1. 论文表格（LaTeX 格式，可直接 \input）

### Table 1：主实验结果
行 = 全部方法（9 行），列按二类分组：
  Personality Traits | Interview Performance | Macro Avg
  Open. Cons. Ext. Agr. Neu. Ovr. | Int.S. Ans.S. Spk.S. Conf. Fac.E. Ovr.P. | MSE↓ r↑
每个单元格 = mean ± std (5 seeds)。最佳结果加粗。统计显著性标注。

### Table 2：消融实验结果
按消融维度分 panel：
  (a) 推理阶段消融 (b) 模态消融 (c) Projection 消融 (d) LLM 消融

### Table 3：计算成本与推理质量
方法 | Token/视频 | 成本/视频 | 推理时间 | 人类可读性评分

## 2. 定性示例（论文 Figure）

选择 3 个代表性视频（高分/中分/低分各一），展示：
  - CueCoT 完整三阶段推理链
  - Generic CoT 的推理链（对比）
  - Ground truth label
  - 标注 CueCoT 推理链中的 evidence grounding 优势

输出为 markdown 文件，后续可转为论文中的 figure。

## 3. 人类评估协议

### 标注设计
- 随机抽取 50 个 test 视频
- 每个视频展示两种推理链：CueCoT（三阶段 CoT）和 Generic CoT（"think step by step"）
- 标注者不知晓推理链来源（盲评），随机排列顺序
- 3 个标注者独立评分

### 评分维度（1-5 Likert）
  1. **Factual Grounding**：推理链是否基于具体可观察的行为事实？
  2. **Logical Coherence**：从证据到结论的推理是否逻辑连贯？
  3. **Overall Trustworthiness**：你多大程度上信任这个评估结论？

### 统计报告
- Krippendorff's α（inter-rater agreement），报告 > 0.67 视为可接受
- CueCoT vs Generic CoT 的配对 t 检验（每维度）
- 均值 ± SD，效应量（Cohen's d）

### 输出
- results/human_eval/annotation_guide.md（标注指南）
- results/human_eval/annotation_template.csv（标注模板）
- results/human_eval/results.json（统计结果）

## 4. AAAI 2027 Reproducibility Checklist

填写 AAAI 2027 要求的可复现性检查表（通常在论文提交系统中）。
在 results/reproducibility_checklist.tex 中生成 LaTeX 源码。

至少包含：
  - 使用的库及版本（MediaPipe, OpenCV, librosa, openai, PyTorch, sklearn）
  - 超参数搜索空间和选择方法
  - 评估指标的准确定义
  - 数据预处理步骤
  - 计算资源（GPU 型号、API 版本）
  - 代码和数据可用性声明

## 5. 架构图

生成 Figure 1（CueCoT 架构概览图）的需求描述文件：
  results/figure_architecture_spec.md

详细描述每个组件、位置、连接关系、标注文字，供后续用 TikZ 或
matplotlib 绘制。

## 6. 论文补全

基于实验产生的实际数字，更新 CueCoT.tex 中以下部分：
  - Abstract 中的定量声明（替换 "consistent improvements" 为实际数字）
  - Section 5.2 Main Results（替换 Placeholder 为实际 Table 1）
  - Section 5.3 Ablation Study（替换 Placeholder 为实际 Table 2）
  - Section 5.4 Qualitative Analysis（替换 TODO 为实际示例）
  - 确认所有交叉引用（Section 引用、Table 引用）正确
```

---

## 质量检查清单（每个阶段完成后自查）

- [ ] 所有数值结果有 mean ± std（5 seeds）
- [ ] 与最佳 baseline 有统计显著性检验（p 值 + CI）
- [ ] 所有超参数已记录并可追溯
- [ ] 数据划分无用户泄露（已验证 user_no 不跨 split）
- [ ] LLM 输出有解析容错，无静默丢弃
- [ ] 人类评估有 inter-rater agreement
- [ ] Reproducibility Checklist 可完整填写
- [ ] 消融实验覆盖所有声称的创新点
- [ ] 论文中的所有定量声明有实验结果支撑
- [ ] 定性示例有代表性（不是 cherry-pick 最好看的）

---

## 附录：常见坑与解决方案

### A1. MediaPipe 初始化开销
每个进程单独初始化 MediaPipe FaceMesh/Pose/Hands。每个进程初始化一次 → 处理该进程分配的所有视频。

### A2. 视频无面部检测
约 5-10% 视频部分帧检测不到人脸。记录 skipped_frame_count，>50% 帧无面部则标记为 low_quality。在 error_analysis 中单独统计该子集表现。

### A3. RecruitView 分数分布
分数已标准化（均值≈0）。LLM 输出需在 prompt 中明确 scale 约束，否则 LLM 倾向输出 0-100 或 1-10。

### A4. GPT-4o API 限速
低 tier 账号 RPM 约 500/min。使用 tenacity 库做指数退避重试（max 5 retries）。

### A5. 成本预估
- 特征提取：仅计算资源（已跑完）
- GPT-4o CoT full：~2000 tokens/视频 × 3 stages × 2011 videos ≈ 12M tokens ≈ $60-120
- GPT-4o-mini（消融用）：约 $3-6
- 建议先用 GPT-4o-mini 调试 prompt，确认格式正确后切 GPT-4o

### A6. 可复现性
所有随机种子固定。所有超参数在代码顶部集中定义（不要隐藏在 argparse 默认值中）。生成 requirements_frozen.txt（pip freeze）。

### A7. CueCoT 分数不在合理范围
如果 LLM 输出分数与 ground truth 的 scale 不匹配，在 prompt 中加 normalization 示例，或在后处理中用 train set 的均值/标准差做线性校准（z-score normalization）。后者需要在论文中说明。

### A8. 逐帧手势特征与面部特征帧数不一致
面部和手势在 extract_features.py 中逐帧同步提取，帧数自然一致。如果偶尔不一致（某帧只有面部没手势），aggregate 时处理：以面部帧索引为基准，缺失手势帧用前一有效帧填充（forward fill）。

# 时长混淆审计

回答一个问题：**现有行为特征的表现，有多少是「视频时长」伪装的？**

起因：RecruitView 的 1410 维特征里，最强的那批（`head_pitch_volatility_volatility`、
`au9_nose_wrinkle_change_rate_std` …）对全部 12 个目标给出几乎相同的 |r|。
同 7 个特征同时"预测"开放性、尽责性和面试表现，这是混淆变量的典型特征，不是特征有效。

## 跑法

全部用 conda 环境 `jingxin`（`~/huihui` 缺 scikit-learn / scipy）：

```bash
PY=~/miniconda3/envs/jingxin/bin/python
cd ~/jingxin/experiments/duration_audit

$PY audit_recruitview.py                        # 约 3–5 分钟（1410 维 × 5 折 RF）
$PY audit_mit.py                                # 约 1 分钟
$PY audit_first_impressions.py --n 60           # 约 1 分钟，--n 可调到 300
```

只读，不改任何现有文件。各自把结果写到同目录的 `results_*.json`。

## 结果（2026-09-21 实测）

| | **RecruitView** | **MIT Interview** | **First Impressions** |
|---|---|---|---|
| 特征来源 | 自动派生，1410 维 | 手挑一阶，272 维 | 无（只有原始视频） |
| 死特征（零方差） | **169 / 1410（12.0%）** | 1 / 272（0.4%） | — |
| 有效维度（90% 方差） | 163 维（11.6%） | 45 维（16.5%） | — |
| 时长平均 \|r\| | **0.322** | 0.142 | **0.037** |
| 最强特征控制时长后 | **缩水 54%**（0.351→0.162） | 缩水 2%（0.524→0.515） | — |
| 模型：仅时长 | **0.256** | 0.135 | — |
| 模型：全部特征 | 0.393 | 0.408 | — |
| **净增益** | **+0.137** | **+0.273** | — |
| 时长跨度 | short/medium/long ≈ 4 倍 | 132s ~ 1053s（8 倍） | **15.313s ± 0.008s（0.05%）** |

> RecruitView 用 5 折 **user-level** 分组 CV；MIT 用 **person-level**（pre/post 同折）。
> 都是 RandomForest(200, min_samples_leaf=5/3)，指标为 macro Pearson r。

## 三条结论

**1. RecruitView 那条线被时长严重污染。**
`voice.duration_sec` 一个变量就拿到 macro r = 0.256，而全部 1410 维拿到 0.393
—— **时长贡献了 65%**。控制时长后，最强特征的相关缩水 54%，
openness / conscientiousness 甚至**符号翻负**。

**2. MIT 那条线是干净的。**
控制长度后最强特征几乎不动（缩水 2%），仅长度只有 0.135 而全特征 0.408。
所以「MIT 上行为 ≫ 文本」这个结论**站得住**，不受时长混淆影响。

**3. 差别不在维度多少，在特征怎么造。**
MIT 的 272 维是**手挑的一阶可解释量**（pitch / 能量 / 共振峰 / 停顿 / 面部角度），
RecruitView 的 1410 维是**同一批信号的自动二阶派生**
（每个 AU 都有 mean / std / trend / volatility / change_rate / start_avg / mid_avg / end_avg …）。
`_volatility`、`_std`、`_change_rate` 这类统计量**随时长单调增长**，
所以它们主要是在测"视频有多长"。

**FI 是唯一天然对照**：片段长度被固定（变异系数 0.05%），与标注相关仅 0.037。

## 修复方向

1. **所有 `_volatility` / `_std` / `_change_rate` / `_count` 类特征按秒归一化**（除以时长或帧数）
2. **时长作为显式元数据存下来**（现在它藏在 `voice.duration_sec` 里）
3. **砍掉 169 个死特征**，并对高度共线的做降维（163 维就够 90% 方差）
4. **转向一阶可解释特征**——MIT 那套（36 面部 + 59 韵律 + 5 微笑）就是现成模板
5. 做模型时**把时长作为协变量**，或至少报告"仅时长"基线作对照

不修这条，系统给一个人打高分可能只是因为他回答得久。

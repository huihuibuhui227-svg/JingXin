# 未决 Minor / deferred 清单 —— 交最终全分支审查 triage

来源:各任务的任务审查与复审里**未进入修复循环**的 Minor/Out-of-Scope 项,加上控制器的独立发现。
用途:最终审查逐条判定「合并前必须修」还是「可以留到 M1–M3」。完整上下文见同目录 `progress.md`。

## 一、报告层代码(Task 6 / Task 7)

| # | 位置 | 内容 | 判级(审查者) |
|---|---|---|---|
| 1 | `report_frontend/report_generator.py`(原 `_get_percentile_badge` 附近行为) | **缺口重复渲染 (e) 无 pytest 守卫** —— 把顶层「证据缺口」段加回去不会让任何测试变红;当初只要求"生成报告核对" | Minor,但**性质无回归保护** |
| 2 | `report_frontend/research_mapper.py:261` | `result["evidence_gaps"]`(顶层并集)在报告层**已无消费方**(仅测试断言它) | Minor |
| 3 | 同上 | **真实数据只走短接路径**;出分路径的报告渲染只由 `_MIXED` fixture 覆盖 | Minor |
| 4 | `report_frontend/feature_engine.py:17` | 类 docstring 仍写「心理特征提取引擎 (最终增强版·科研数据专用)」,是禁止词全量扫描里**唯一非注释**的「科研/心理」自述。不进输出、无禁止词,故不违约束 | Minor |
| 5 | **`templates/dashboard.html:65-66`** | 生成报告的按钮仍叫「📑 生成综合判推报告」/「整合所有数据,生成 HTML 深度评估文书」 —— **用户可见**,与它现在真正生成的东西(无判推、无评估)矛盾。`templates/` 在范围内 | Minor,**但属用户可见的过度承诺** |
| 6 | `tests/test_report_layer.py` | 扫描只 glob `*.py`,`templates/*.html` 与 `evidence_thresholds.json` 里的字符串**无法让它变红**(今天靠人工 grep 兜) | Minor |
| 7 | `tests/test_report_layer.py` | 反空转守卫只看总量(`>= 5`,实际 9),单个根目录整体消失仍可能过关 | Minor |
| 8 | `report_frontend/research_mapper.py` | 覆盖率计数靠 `int()` 解析展示串 `matched_indicators`(今天正确且 fail-fast,但计算耦合了展示格式) | Minor |
| 9 | `report_frontend/visualizer.py` | 本任务删除动作留下的死状态:`charts['gaze']` 无消费方;`positive_factors`/`negative_factors` 失去唯一读者 | Minor |
| 10 | `report_frontend/visualizer.py:142` | `create_gaze_plot_from_df(self, df_face: pd.DataFrame)` 忽略入参、且调用方可能传 `None`,标注不准 | Minor(纯标注) |

⚠️ **第 4 项附带一条纪律**:Task 7 实现者在报告 §8.2 里用「三处「心理/科研」都挂在类名上」解释为何不改 —— **该理由不成立**(类名是 ASCII `PsychologicalFeatureEngine`,那三处是 docstring 文本)。审查者明确要求**不得以该理由关掉该项**。

## 二、设计文档(Task 8)

Task 8 的 2 条 Important + 4 条 Minor 已**全部折进其修复轮 1**(Ruling 37),此处不重复。若修复轮后复审仍有遗留,记在 `progress.md` 的 Task 8 段落。

## 三、控制器独立发现(不属于任何任务)

| # | 内容 | 状态 |
|---|---|---|
| 11 | **spec §5.3 / §7.3 与 `docs/下一步.md` 都写「21 个指标槽」,实测代码是 20** —— merge base `07ed73d` = 21,`36f1f87`(Task 3 首个提交)起 = 20,差额是 Task 3 删掉的**重复计数槽**(comm 维度权重和 1.4 → 1.0)。**报告输出的「20」与代码一致,是文档陈旧** | 待文档修订 |
| 12 | **spec §6 第 2 条(真实会话回归:断言不再出现「置信度:高」)只有人工证据** —— 单元层有 `test_high_is_unreachable_this_round`(穷举 3600 组返「高」0 次)+ `test_confidence_can_be_none_and_low`,但**"对 `data/logs/` 每条真实日志跑完整链路"这一步没固化成测试** | 待补测试或明确接受人工 |
| 13 | **`.pyc` 卫生**(`docs/下一步.md` §5 第 3 项):64 个 `.pyc` 仍被 git 跟踪且已从磁盘删除;`.pytest_cache/` 未进 `.gitignore`。**属仓库卫生改动,未获使用者明示授权,控制器不擅动** | **需使用者决定** |
| 14 | **把空断言扫描(`sys.settrace` 记录已执行行 vs AST 断言行)固化成常驻测试**(§5 第 4 项)—— 本轮该缺陷形态出现 **8 次**,目前靠人工跑脚本 | **需使用者拍板**(建议做) |
| 15 | 封停优先(G4 提到 G2 前,§5 第 2 项)—— 现在"既恒定又封停"的槽显示「本次会话内无变化」而非「该指标本轮停用」。改它要重排关卡顺序,而 Task 2 的测试正钉着当前顺序 | 默认**不改**(§5 默认) |
| 16 | **`.superpowers/` 账本被 git 忽略**,合并后可能被清掉(§5 第 5 项) | 建议手动留档 |
| 17 | 增补 §1.2 的措辞「理由替换为**剩下的三条**」与设计文档 §4.4 的**四个理由**对不上(Ruling 36 裁定按 brief 保留四条) | 待一句话对齐 |

## 四、已经在本轮修掉、无需再看的

- 禁止词经 mapper 的 `display_name`/`human_name`/`description`/`inference_template` 进入完整报告 → **Task 7 已修**,并有全 AST 扫描测试守着。
- 三条叙事层测试的空断言、`symmetry_score` 断言不可证伪 → **Task 6 Step 8 已修**,复审逐条 ADDRESSED。

## 五、Task 8 修复轮上报的新增未决项

| # | 内容 | 归属 |
|---|---|---|
| 18 | **数据集派生的 L1 阈值是否落入许可的「models trained on it」(Ruling 38)** —— §5.1 规则 1/5、§5.2 的 τ(x)、`:174` 的 `au26` p1/p99,尤其 `:295`「`au12` 的 p90 取自常模人群,上线时固定使用」。控制器按**保守读法**裁定"是"(在数据集分布上拟合的参数 = 数据集派生物),但**动作属 M5 的阈值设计**,不是 ① 的范围 | **需使用者做 M5 范围决策**;建议在 §10 或 §5 加一条限制条目 |
| 19 | §10 没有"许可/伦理限制"条目;§1.3 的离线实验未标许可 | 文档完整性 |
| 20 | §6.6 示例的 `basis` 列了 `response_latency_sec`,而 §11.1 写明它到 M1 才可用 —— 示例用了尚不可用的输入 | 文档示例 |
| 21 | 增补 §1.2 的「理由替换为**剩下的三条**」与设计文档 §4.4 的四个理由措辞对不上(Ruling 36 裁定按 brief 保留四条) | 一句话对齐(与 #17 同条) |

⚠️ 已澄清、**不需改**的一项:§6.6 示例里的 `"logical_thinking"` **不是**"改名前的旧键" —— Task 7 改的是展示名(`name`),内部键名未动,`logical_thinking` 是稳定标识符(Ruling 39)。

## Task 8: 修订设计文档的三处前提(spec §1)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md`

**Interfaces:**
- Consumes: 无
- Produces: 无(纯文档)

这三处不涉及代码,但与 ① 同期完成,否则设计文档会与增补矛盾。

- [ ] **Step 1: 修订 §6.3 阶段 1**

把 `| 1 | 三个数据集标定(**先上线**) | 低 |` 这一行替换为:数据集标定**不上线**,并写入 spec §1.1 的两条可选路径(a:有自有标注前 L2 不上线;b:数据集线严格限定为离线研究结论)。

同时在 §6.3 表格上方加一句:

> ⚠️ RecruitView 许可(CC BY-NC 4.0)明文禁止将本数据集**及在其上训练的模型**用于真实招聘、雇佣筛选或**心理画像**。故"数据集标定 → 先上线"这一组合被移除。详见增补 §1.1。

- [ ] **Step 2: 修订 §4.4 的四个理由**

删除第 1 条("声学特征可跨语言迁移,语言特征不能")中的该前提,替换为:

> 1. **跨语言可迁移性不对称(仅适用于非 f0 类特征)** —— 语言特征不能跨语言迁移。注意:f0 类特征是**例外**,普通话的词汇声调压缩了情感语音中的音高变异,故 f0 类特征同样不可跨语言直接搬运(增补 §1.2)。

保留其余三条(失败模式不同 / 隐私等级不同 / 时间线不同)。

- [ ] **Step 3: §6.2 补伦理审查**

在 §6.2 的"具体量表版本与授权需确认后使用"之后加:

> **⚠️ 强制前置项:** 按《科技伦理审查办法(试行)》第 2 条第(一)项,以人为研究参与者(含"利用个人信息数据")的科技活动**必须通过科技伦理审查**。采集候选人自评量表用于标定落入该条,2023-12-01 已施行。此项与量表授权并列,不得省略。

- [ ] **Step 4: 提交**

```bash
git add docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md
git commit -m "docs: 按心理测量学审查修订设计文档三处前提"
```

---

## 收尾:回归与验收

- [ ] **Step 1: 全量测试**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 2: 采集层回归防线必须仍然通过**

Run: `~/miniconda3/envs/jingxin/bin/python experiments/duration_audit/reaggregate_normalized.py --stats legacy --verify-legacy`
Expected:0/2835510 格差异。**① 只改报告层,这条防线若被打破说明改错了范围。**

- [ ] **Step 3: 真实会话端到端**

按 Task 6 Step 6 跑一遍,人工确认报告内容。

- [ ] **Step 4: 与 spec 逐条对照**

打开 spec §6 的 7 条验证方式,逐条确认有对应测试或人工步骤。

# report_frontend/report_generator.py

import os
import re
import webbrowser
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from .data_loader import LogDataLoader
from . import l1_mapper
from .l1_report import build_l1_report
from .visualizer import ReportVisualizer


#: 维度数 → 中文数字。只到六维就够(当前最多五维,消失一维后四维);
#: 表外的数原样印阿拉伯数字 —— 宁可印 `7 个维度`,不许印一个错的中文数。
_CN_NUM = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}


def _fmt_num(v) -> str:
    """实测值 → 给人读的字符串。**小数位数按量级选,不一律两位。**

    实测产出过 `3.4175334323922733`(float64 的内存映像)。前端
    `L1IndicatorTable` 的 `fmtValue` 早就有一条按量级的规则,HTML 报告是
    同一个数的另一个出口,不该有两套精度:

    * ≥1000 取整;≥1 三位;≥0.001 四位;更小用科学计数。

    ⚠️ **只在渲染处收**。载荷里的 `raw_value` 保持原始 float ——
    前端要拿它排序/比较,在那里四舍五入就是丢测量结果。
    钉子:`tests/test_report_numbers_and_reasons_are_readable.py`。
    """
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:                       # NaN
        return "—"
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1000:
        return f"{x:.0f}"
    if a >= 1:
        return f"{x:.3f}"
    if a >= 0.001:
        return f"{x:.4f}"
    return f"{x:.2e}"


def _interval_text(interval) -> str:
    """本场会话内观测区间。拿不到就说清是缺什么,而不是留空。

    区间是 `原始值 ± 会话内标准差`(与原始值同量纲、来自本场会话自己的测量),
    不是人群中位置 —— spec §5.5 禁止任何暗示人群位置的区间。
    """
    if not interval:
        return "未采集到会话内变异信息"
    return f"{interval[0]} – {interval[1]}"


# 模态键 → 报告里给人看的名字(键名来自 data_loader 的统一化,见那边的 key 映射)。
_MODALITY_LABELS = {
    "face": "面部",
    "gesture": "手势",
    "voice_interview": "语音（面试）",
    "voice_research": "语音（科研）",
}

#: 日志键 → L1 的**族模态名**(`l1_quality.FAMILY_MODALITY` 用的是后者)。
#: 两者不同名是因为日志按"面试/科研"分文件,而 L1 的 V 族不分 —— 同一套指标。
#: 这张表只用于把载荷里的 `modality_summary` 对上号,不做别的判断。
_LOG_KEY_TO_MODALITY = {
    "voice_interview": "voice",
    "voice_research": "voice",
}


def sources_disclosure(sources: Dict[str, Dict], target: Optional[str] = None,
                       modality_summary: Optional[Dict[str, Dict]] = None) -> str:
    """披露「本报告描述的是哪一场、每个模态进来了没有」(M2 spec §5.2)。

    输入是 `LogDataLoader.selected_sessions`,值是三态之一:
    `loaded` / `unreadable`(文件在但读不出)/ `missing`(本场没有);
    另有一个特殊键 `none_bucket`(没带 session_id 的行,不进聚合但要说一声)。

    **为什么"未读到数据"必须显式出面**:I4 的实现只列真正读出来的模态,于是读不出来的
    那些**静默消失**,而报告还写着「以上模态来自同一场会话」—— 2026-09-24 实测:语音整个
    没了,读者却以为一切正常(spec §3.3)。

    同场/不同场那句话**已取消**:按 id 选之后不存在"不同场"这个状态,真正要报的是
    "哪几个模态没进来、为什么"。

    `target` 是"本报告以哪一场为准",由调用方给(`LogDataLoader.target_session` /
    实时路径的入参,第 17 条)。**为什么不从 items 里推第一个**:只有 NONE 桶、或连
    NONE 桶都没有时,items 是空的,推不出来 —— 而那正是最需要把话说清楚的情形。
    没给且 items 非空时退回"取第一个"(兼容直接调用本函数的调用方与测试)。

    ## `modality_summary`(2026-10-25 补):日志三态**不是全部事实**

    三态描述的是**日志文件**。语音还有第二个数据源:整场录音逐帧重算
    (`voice_series`),不经日志。实测场 `20260928_180131_8102` 语音日志 0 行,
    报告头说「未读到数据(…若本场确实答过题,那才是文件损坏)」,而正文里
    V 族 12 条有 10 条有值 —— 同一个模态,两个出口说了相反的话。

    载荷 (`l1.modality_summary`) 给出每个模态到底算出了几条,这里**照抄**:
    有值 ⟹ 说「N/M 条有值」,不再提"损坏";一条都没有 ⟹ 保留原措辞(那是真的)。
    渲染层不自己数 —— 它拿不到族与模态的归属(`l1_quality.FAMILY_MODALITY`)。
    """
    none_bucket = sources.get("none_bucket")
    items = {k: v for k, v in sources.items() if k != "none_bucket"}
    if target is None and items:
        target = next(iter(items.values()))["session_id"]

    summary = modality_summary or {}

    def _produced(key: str) -> Optional[Dict]:
        """这个模态**算出了几条** —— 日志键 `voice_interview` 归到 `voice` 这一族。

        没登记就返回 `None`(不猜)。判据只有这一处,两个分支共用。
        """
        fam = _LOG_KEY_TO_MODALITY.get(key, key)
        return summary.get(fam)

    def _produced_line(label: str, key: str) -> Optional[str]:
        got = _produced(key)
        if not got or not got.get("n_measured"):
            return None
        note = f"（{got['source_note']}）" if got.get("source_note") else ""
        return (f"{label} · <strong>日志没有数据，但这个模态算出了 "
                f"{got['n_measured']} / {got['n_total']} 条指标</strong>{note}")

    lines = (["本场会话：<strong>（无 —— 本场没有任何日志）</strong>"] if target is None
             else [f"本场会话：<strong>{target}</strong>"])
    if not items:
        lines.append("本报告没有装配任何模态日志。")

    for key, info in sorted(items.items()):
        label = _MODALITY_LABELS.get(key, key)
        if info["status"] == "loaded":
            lines.append(f"{label} · 已读入（{info['rows']} 行）")
        elif info["status"] == "unreadable":
            # ⚠️ 措辞别冤枉文件。此前这句写的是"文件在，但读不出来：空文件或损坏",
            #    而最常见的情形**根本不是损坏** —— 是这一场压根没有提交过回答
            #    (从「实时分析」页录的那些就是:那条路没有题库、不提交回答,
            #    语音模态因此结构上为空,交接日志 §8.1-4 记过)。两件事的处置完全不同:
            #    一个是"设计如此",一个是"去查文件"。分不开就会让人去追一个不存在的故障。
            #
            # ⚠️ 更要紧的是:日志空 ≠ 这个模态没有量。语音可由整场录音重算 ——
            #    有值时先说这件事,别让读者带着"文件可能损坏"的印象往下读。
            got = _produced_line(label, key)
            if got:
                lines.append(got)
            else:
                lines.append(f'<strong style="color:#A23B72;">{label} · 未读到数据</strong>'
                             f'（日志文件在，但一行数据都没有。常见原因是本场'
                             f'<strong>没有提交过逐题回答</strong> —— 例如从「实时分析」页录的，'
                             f'那条路没有题库；若本场确实答过题，那才是文件损坏。）')
        else:
            # 同上:盘上连日志文件都没有,但指标可能由第二个数据源算出来了。
            got = _produced_line(label, key)
            if got:
                lines.append(got)
            else:
                lines.append(f"{label} · 缺失（本场没有这个模态的日志）")

    if none_bucket:
        lines.append(f'<em>另有 NONE 桶 {none_bucket["rows"]} 行 —— '
                     f'那些请求没带 session_id，不属于本场，未参与计算。</em>')

    return "<br>".join(lines)


def live_sources(session_id: str, data: Dict[str, Any]) -> Dict[str, Dict]:
    """实时路径的数据来源:模态 → 三态条目(与批处理路径**同一个形状**)。

    实时路径是按**这一个 `session_id`** 去内存里取的(不像批处理要在磁盘上选文件),
    所以凡取到数据的模态都是 `loaded`。

    ⚠️ 形状必须与批处理一致。这条曾经漏改(2026-09-24 复审 C1):旧形状是
    `{k: session_id}`,而 `sources_disclosure` 会做 `info["status"]` → `TypeError`,
    **而面板照样报 success** —— 又一个静默失败,且全仓没有测试碰过这条路径。
    """
    return {k: {"session_id": session_id, "status": "loaded", "rows": len(v)}
            for k, v in data.items()}


class ReportGenerator:
    """
    行为观测报告生成器

    报告只陈述本次实际测到了什么:每个维度按 spec §5.4 的「值 + 有效样本量 + 置信度 +
    evidence_gaps」渲染,聚合只给「区间 + 置信度 + 依据」(spec §5.6)。
    **不产出对候选人的评分或评级** —— 未标定标尺上的复合点分与五档评语一律不渲染。
    不解读、不推断、不做形容词修饰。
    """

    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_report(self, session_id: Optional[str] = None) -> str:
        """从磁盘 CSV 文件生成评估报告（批处理模式）。

        `session_id`:本报告要描述的那一场(M2)。**不给就取最新一场**,并在报告头写明是谁
        —— 这个参数在 M2 之前是"收下就丢"的。
        """
        print("\n" + "=" * 70)
        print("🚀 启动 JingXin 面试行为观测报告生成系统 (批量模式)")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"Research_Assessment_Report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)

        try:
            loader = LogDataLoader()
            data = loader.get_fused_latest_data(session_id)
            if not data:
                # 本场一个模态都没读到:**照样出报告**(spec §6 行 1/2 / 第 17 条)——
                # 头里逐条列出缺什么,而不是让面板显示"什么都没发生"。
                # 实测:空 data 下 L1 给 覆盖 0/14、逐维缺口照常列出、图表照常生成。
                print("   ⚠️  本场没有任何可用日志,仍生成一份只含缺口的报告")

            # ★ 与 `/api/report/structured` **同一个引擎**(2026-10-25)。
            #
            #   2026-10-18 前端那条端点已切到 L1,而这里一直留在旧引擎
            #   (`PsychologicalFeatureEngine` → `ResearchCapabilityMapper`)。
            #   后果:同一场会话在界面上给出**两个互相矛盾的数** —— 前端报告页读
            #   structured(2026-10-25 实测 7/14),而这里产出的 HTML 说 11/14。
            #   差的那 4 个槽正是旧引擎把「手掌张开度 / 肩部放松度 / 眼神接触」算成
            #   "填上了",而 L1 判定**当前管线根本不测**(见 `docs/下一步.md` 第 29 条)。
            #   两份东西在「历史报告」列表里并排,谁也看不出它们量的是两套口径。
            #
            #   对照该在**数据层**做(`?engine=legacy` 仍在),不该让交付物分叉。
            #   钉子:`tests/test_html_report_uses_the_l1_engine.py`。
            result = build_l1_report(loader.target_session, logs=data)

            viz = ReportVisualizer(output_dir=self.output_dir)
            chart_paths = viz.generate_all_charts(result, df_face=data.get('face'))
            static_images = self._scan_static_images()

            # `features` 是旧引擎的中间产物;L1 载荷自带逐槽值,渲染层不再需要它。
            html_content = self._build_html_report(result, chart_paths, {}, data,
                                                   static_images,
                                                   sources=loader.selected_sessions,
                                                   target=loader.target_session)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            print(f"❌ 错误：{e}")
            return ""

        # 报告**已经落盘**。打开浏览器是"顺手"的副作用 —— 它失败不许影响返回值:
        # 退出码是本轮新引入的(第 17 条),一旦从这里抛出去,面板会把"报告其实写好了"
        # 报成「任务失败」(假警报)。
        try:
            webbrowser.open('file://' + os.path.realpath(report_path))
        except Exception as e:
            print(f"   ⚠️  报告已生成,但打开浏览器失败：{e}")
        print(f"\n✅ 报告已生成：{report_path}")
        return report_path

    def generate_report_live(self, session_id: str) -> str:
        """从运行中的 API 服务获取实时内存数据，生成评估报告（实时模式）"""
        print("\n" + "=" * 70)
        print("🚀 启动 JingXin 面试行为观测报告生成系统 (实时模式)")
        print(f"📋 会话 ID: {session_id}")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"Research_Assessment_Report_Live_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)

        try:
            loader = LogDataLoader()
            data = loader.get_live_data(session_id)
            if not data:
                raise ValueError("无法从 API 获取实时数据，请确认三个分析服务 (8000/8001/8002) 已启动且会话存在")
            if 'face' not in data:
                print("   ⚠️  未获取到面部数据，继续使用其他模态生成报告")

            # 与批量路径同一个引擎(理由见 `generate_report` 那段注释)。
            # 实时路径的 `logs` 来自三个服务的 `/summary`,形状与读盘那份相同。
            result = build_l1_report(session_id, logs=data)

            viz = ReportVisualizer(output_dir=self.output_dir)
            chart_paths = viz.generate_all_charts(result, df_face=data.get('face'))
            static_images = self._scan_static_images()

            # 实时路径的模态是**按这一个人的 session_id 去内存里取的**,不存在"每模态取最新"
            # 那种跨场拼接;所以来源就是这一个 id(逐个列出真正取到数据的模态)。
            html_content = self._build_html_report(result, chart_paths, {}, data,
                                                   static_images,
                                                   sources=live_sources(session_id, data),
                                                   target=session_id)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            webbrowser.open('file://' + os.path.realpath(report_path))
            print(f"\n✅ 实时报告已生成并打开：{report_path}")
            return report_path

        except Exception as e:
            print(f"❌ 错误：{e}")
            return ""

    def _scan_static_images(self) -> List[str]:
        images = []
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'trajectory' not in f and 'chart' not in f:
                    images.append(f)
        return images

    def _render_dimension_block(self, dim_key: str, dim: Dict[str, Any]) -> str:
        """渲染单个维度的证据状态。不解读,不推断,不加形容词。

        形态 = spec §5.4 的「整节的正确形态」:值 + 有效样本量 + 置信度 + evidence_gaps。
        出分维度另带 §5.6 要求的提示句「该维度目前无独立效标,仅供行为描述」。
        **不再渲染"归一值"** —— 它是未标定标尺上的点分,与档位标签一样会被读成
        对该维度的评定,而本系统没有标定样本能给这个刻度背书。
        表格给出的"本场会话内观测区间"只由本场测量构成(均值 ± 会话内标准差)。

        每个维度自己带全自己的缺口 —— 出分维度也要列出未过门的槽。
        否则那些缺口只能靠顶层的汇总段兜底,而汇总段的所有条目都已被逐维列表
        覆盖过一遍(evidence_gaps 就是各维缺口的并集),同一批字符串会被打印两次。
        """
        # ⚠️ 分支判据 2026-10-04 由 `dim["score"] is None` 改成 `not dim["evidence_chain"]`
        #    —— payload 里的点分已摘除(见 `research_mapper` 返回处那段注释)。
        #    两者**等价**:`score` 为 None 当且仅当该维一个槽都没过门,而那正是
        #    `evidence_chain` 为空。判据从"有没有分"改成"有没有证据",本来也更贴本仓的主张。
        if not dim["evidence_chain"]:
            gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
            return f"""
            <h3>{dim['display_name']}</h3>
            <p><strong>证据不足</strong> —— 本次未采集到足以评估该行为线索的有效样本。</p>
            <ul>{gaps}</ul>
            """

        rows = "".join(
            f"<tr><td>{e['human_name']}</td><td>{_fmt_num(e['raw_value'])}</td>"
            f"<td>{_interval_text(e.get('observed_interval'))}</td>"
            f"<td>{e.get('n_valid', '—')}</td></tr>"
            for e in dim["evidence_chain"]
        )
        gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
        gaps_block = f"<p>未过门的指标：</p><ul>{gaps}</ul>" if gaps else ""
        return f"""
        <h3>{dim['display_name']}</h3>
        <p>过门指标 {dim['matched_indicators']}；置信度：<strong>{dim['confidence']}</strong>。
        （该维度目前无独立效标，仅供行为描述）</p>
        <table><thead><tr><th>指标</th><th>原始值</th><th>本场会话内观测区间</th>
        <th>有效样本量</th></tr></thead>
        <tbody>{rows}</tbody></table>
        {gaps_block}
        """

    def _generate_deep_text_analysis(self, features: Dict[str, Any], result: Dict[str, Any]) -> str:
        """只陈述本次实际测到了什么,不做任何解读、推断与形容词修饰。

        每个维度按 spec §5.4 的「整节的正确形态」渲染,出分维度带 §5.6 的"无独立效标"
        提示。本函数**不再输出聚合摘要**:它曾印「综合科研潜力评分为 X 分,评级为 Y」,
        后改印「综合行为观测摘要:{档位}(置信度上限:…)」—— 两者都是把未标定标尺上的
        点分与五档评语当对人的评定(实测报告头渲染出「0.0 / 综合行为观测评分 / 待提升」)。
        聚合呈现移入报告头部的覆盖卡,只给「区间 + 置信度 + 依据」(spec §5.6)。

        参数 features 保留是为了调用方签名稳定;本函数不再从原始特征里另取默认值。
        """
        parts = []
        for dim_key, dim in result["dimensions"].items():
            parts.append(self._render_dimension_block(dim_key, dim))
        # 原本此处另有一段顶层「证据缺口」汇总,现已删除 —— 它的每一条都来自
        # result["evidence_gaps"](各维缺口的并集),而逐维块已经把各自的缺口列全,
        # 于是同一批字符串会在报告里出现两遍。缺口现在只有逐维这一处来源(带维度归属)。
        return "".join(parts)

    def _render_vanished_dimensions(self, result: Dict[str, Any]) -> str:
        """整维消失的那一维要**说出来** —— 前端说了,交付物也得说。

        L1 的构念对账判定 `行为表现活跃度` 的三个槽没有同构念的 L1 指标 ⟹
        它整个不进 `dimensions`,而进 `vanished_dimensions`。前端 `ReportViewer`
        为此专门加了一块卡片,注释原话是「不渲染的话报告会**悄悄少一块**,
        而少的那块恰好是『测不到的东西』—— 那是最该说出来的部分」。

        但本文件原先只遍历 `result["dimensions"]` ⟹ **HTML 报告里那一维凭空消失**。
        同一场会话:界面上说"这一维整维未出",下载的 HTML 说"只有四个维度"。

        ⚠️ 措辞必须是「本层不测」而不是「本场没测到」—— 后者让人等下一场,
        前者才是实话。逐槽理由走 `l1_mapper.drop_reason`(人话那层),
        槽名走 `slot_label`,内部键不出现在交付物里。
        钉子:`tests/test_html_report_shows_vanished_dimensions.py`。
        """
        vanished = result.get("vanished_dimensions") or []
        if not vanished:
            return ""
        items = "".join(
            f"<li><strong>{v['display_name']}</strong>"
            f"(声明了 {v['n_slots']} 个指标槽,全部没有同构念的 L1 指标)"
            f"<ul>" + "".join(
                f"<li>{l1_mapper.slot_label(d['slot'])}:本层不测这一条 —— "
                f"{l1_mapper.drop_reason(d['slot'])}</li>"
                for d in v.get("dropped", [])) + "</ul></li>"
            for v in vanished)
        return f"""
        <div id="vanished-dimensions">
            <h3>整维未出(当前管线没有对应的可测量)</h3>
            <p>下面这一维<strong>不是本场没测到</strong>,是它要量的东西当前管线都不测 ——
            再采一百次也不会有,别等它。</p>
            <ul>{items}</ul>
        </div>
        """

    def _coverage_heading(self, result: Dict[str, Any]) -> str:
        """「本次观测覆盖」那张雷达图的标题 —— **维度数从载荷算,不是字面量**。

        这里原先写死 `📊 五维证据覆盖`。旧引擎给满五维,所以那句一直对;
        L1 引擎下 `行为表现活跃度` 整维消失 ⟹ 正文里只有**四个**维度,
        而标题还写着「五维」—— 同一份报告自己跟自己矛盾,读者会以为少了一维。

        ⚠️ 数的是 `dimensions`,**不含** `vanished_dimensions`:消失的那一维
        在正文里另有一块说明(「本层不测」),把它算进"本次观测"会让读者
        以为它出了值。钉子:`tests/test_the_report_stops_saying_wrong_things.py`。
        """
        n = len(result.get("dimensions") or {})
        return f"📊 {_CN_NUM.get(n, n)}维证据覆盖（{n} 个维度）"

    def _render_l1_audit_text(self, result: Dict[str, Any]) -> str:
        """把载荷里那四组「这份报告能说什么」的前提印出来。

        前端 `L1IndicatorTable` 全都渲染了(它的注释原话:放在表格上方而不是折叠里,
        因为这些是"这份报告能说什么"的前提),而 HTML 报告**一个都没印**:

        | 载荷字段 | 它回答的问题 | 不印的后果 |
        |---|---|---|
        | `l1.gaps` | 本场素材层面的限制(拼接、没有提问台账、没有 ASR) | 读者把只覆盖最后 7 分钟的数当成整场 |
        | `l1.comparability_notes` | 哪些槽名与旧报告相同但统计量换了 | 读者把两份报告的数相减 |
        | `l1.low_confidence_notes` | 置信度为什么只有"低" | 读者无从判断这个数能不能用 |
        | `l1.windows` | 整场一个窗口,还是逐回答一个窗口 | 两个口径的 F/G 值被当成同一件事 |

        ⚠️ 全部来自载荷,**渲染层不另写一份**(否则迟早分叉)。
        四组都空时不印任何标题 —— 空标题会被读成"这里该有内容但丢了"。
        钉子:`tests/test_html_report_carries_the_audit_text.py`。
        """
        l1 = result.get("l1") or {}

        def _plain(t: str) -> str:
            # 载荷里的这几句是**开发者写的注释体**,带 `**` / 反引号。
            # 前端 `L1IndicatorTable.plain()` 做同一件事(纯文本界面里它们是噪音)。
            return re.sub(r"\*\*|`", "", t)

        def _ul(items) -> str:
            return "<ul>" + "".join(f"<li>{_plain(str(i))}</li>" for i in items) + "</ul>"

        blocks: list[str] = []

        gaps = l1.get("gaps") or []
        if gaps:
            blocks.append(
                "<div class='l1-audit'><h3>本场报告的限制（素材层面，不是算错）</h3>"
                + _ul(gaps) + "</div>")

        # 印**人话版**:审计版(`comparability_notes`)里是「旧槽键 → L1指标名：…」,
        # 那 6 个内部指标名不该出现在交付物正文里(前端同理由 `plain()` 之外还要换表)。
        # 载荷没发人话版时退回审计版 —— 宁可露出英文,也不许把这一条吞掉。
        notes = (l1.get("comparability_notes_plain")
                 or l1.get("comparability_notes") or [])
        if notes:
            blocks.append(
                "<div class='l1-audit'><h3>与旧引擎报告对比时注意</h3>"
                "<p>下列槽名相近，但统计量已更换，两边的数<strong>不可相减</strong>。</p>"
                + _ul(notes) + "</div>")

        low = l1.get("low_confidence_notes") or []
        if low:
            blocks.append(
                "<div class='l1-audit'><h3>置信度为「低」的原因（素材限制，不是算错）</h3>"
                + _ul(low) + "</div>")

        w = l1.get("windows") or {}
        if w.get("granularity"):
            is_answer = w["granularity"] == "answer"
            text = (f"逐回答窗口（{w.get('n_windows', '?')} 个）" if is_answer
                    else "整场一个窗口")
            outside = {k: n for k, n in (w.get("frames_outside") or {}).items()
                       if isinstance(n, (int, float)) and n > 0}
            extra = ""
            if outside:
                extra = ("；不在任何回答窗口内的帧：" +
                         " · ".join(f"{k} {n} 帧" for k, n in outside.items()) +
                         "（这些帧不计入 F/G 的分母）")
            blocks.append(
                f"<div class='l1-audit'><h3>统计口径</h3><p>{text}{extra}</p></div>")

        return "".join(blocks)

    def _build_html_report(self, result: Dict[str, Any], chart_paths: Dict[str, Any],
                           features: Dict[str, Any], data: Dict[str, pd.DataFrame],
                           static_images: List[str],
                           sources: Optional[Dict[str, Dict]] = None,
                           target: Optional[str] = None) -> str:

        def get_chart_iframe(path, height="500"):
            if not path or not os.path.exists(path): return '<div class="placeholder">图表缺失</div>'
            return f'<iframe src="{os.path.basename(path)}" width="100%" height="{height}px" frameborder="0"></iframe>'

        deep_analysis_html = self._generate_deep_text_analysis(features, result)

        # 头部聚合呈现:只讲事实(spec §5.4 / §5.6)。
        # 原分数卡渲染「点分 + 五档评语」(如「0.0 / 综合行为观测评分 / 待提升」)——
        # 那是把未标定标尺上的复合点分当对候选人的评定。改为:覆盖事实 + 依据 +
        # 区间,区间只能来自本场会话自己的测量(§5.5:没有真实常模就不给位置)。
        coverage = result["coverage"]
        basis_items = "".join(
            f"<li>{slot['display_name']} · {slot['indicator']}："
            f"原始值 {_fmt_num(slot['raw_value'])}；"
            f"本场会话内观测区间 {_interval_text(slot['observed_interval'])}"
            f"（与原始值同量纲）；有效样本量 {slot['n_valid']}</li>"
            for slot in coverage["passed_slots"]
        )
        basis_html = (f"<ul>{basis_items}</ul>" if basis_items
                      else "<p>本次会话没有指标通过证据门。</p>")

        # 证据链贡献度图已随评分层摘除(见 `visualizer.generate_all_charts` 里的说明)——
        # 逐槽的诚实呈现是下面那些维度块里的表(原始值 + 区间 + 有效样本量 + 置信度)。
        # `chart_paths['evidence']` 现在恒为空(见 `generate_all_charts`),不再进 HTML。

        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>JingXin 面试行为观测报告</title>
            <style>
                :root {{ --primary: #2E86AB; --bg: #f4f7f6; }}
                body {{ font-family: 'Microsoft YaHei', sans-serif; background: var(--bg); color: #333; margin: 0; padding: 20px; line-height: 1.8; }}
                .container {{ max-width: 1100px; margin: 0 auto; }}
                header {{ text-align: center; padding: 40px; background: linear-gradient(135deg, #2E86AB, #A23B72); color: white; border-radius: 10px; margin-bottom: 30px; }}
                h1 {{ margin: 0; font-size: 2.5em; }}
                .score-board {{ display: flex; gap: 20px; margin: 20px 0; }}
                .score-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; }}
                .cover-number {{ font-size: 3em; font-weight: bold; color: var(--primary); }}
                .card {{ background: white; padding: 30px; margin-bottom: 25px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                h2 {{ border-left: 5px solid var(--primary); padding-left: 15px; color: var(--primary); }}
                h3 {{ color: #444; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                .chart-container {{ margin: 20px 0; border: 1px solid #eee; border-radius: 5px; }}
                @media (max-width: 768px) {{ .score-board {{ flex-direction: column; }} }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>🔬 JingXin 面试行为观测报告</h1>
                    <div>基于多模态行为量的结构化观测</div>
                    <div style="margin-top:10px; font-size:0.9em; opacity:0.8;">
                        {datetime.now().strftime("%Y-%m-%d %H:%M")} | {result['model_metadata']['version']}
                    </div>
                    <!-- 本报告由哪些日志装配(I4):跨场拼接必须让读者看得见 -->
                    <div style="margin-top:8px; font-size:0.85em; opacity:0.95;">
                        {sources_disclosure(sources or {}, target,
                                            modality_summary=(result.get("l1") or {})
                                            .get("modality_summary"))}
                    </div>
                </header>

                <!-- 覆盖与依据:分数与档位已按 spec §5.4 :157-158 / §5.6 停止渲染 -->
                <div class="score-board">
                    <div class="score-card">
                        <div>📊 本次观测覆盖</div>
                        <div class="cover-number">{coverage['n_passed']} / {coverage['n_slots']}</div>
                        <div>个指标槽通过证据门</div>
                        <div style="color:var(--primary); font-weight:bold;">置信度上限：{coverage['confidence_cap']}</div>
                    </div>
                    <div class="score-card" style="flex:2; text-align:left;">
                        <h3 style="margin:0 0 10px 0; border:none;">📝 综合总结</h3>
                        <p style="margin:0;">{result['summary_narrative']}</p>
                        <h3>依据：通过证据门的指标</h3>
                        {basis_html}
                    </div>
                </div>

                <!-- 这份报告能说什么的前提:限制 / 比较陷阱 / 低置信理由 / 统计口径。
                     前端 L1IndicatorTable 早就印了,交付物不该少说这四句。 -->
                {self._render_l1_audit_text(result)}

                <!-- 逐维观测明细:过门指标 + 未过门缺口 -->
                <div class="card">
                    <h2>📑 行为指标观测明细</h2>
                    {self._render_vanished_dimensions(result)}
                    {deep_analysis_html}
                </div>

                <div class="card">
                    <h3>{self._coverage_heading(result)}</h3>
                    <div class="chart-container">{get_chart_iframe(chart_paths.get('radar'), '500')}</div>
                </div>
                <!-- 这里原来还有一行 `<h2>🔍 分维度证据链</h2>` + 逐维度的
                     「证据链贡献度」图(每维一张卡)。图在 2026-10-04 随评分层一起
                     摘除 —— 它们的 x 轴是 `contribution`(折算值 × 槽权重),即
                     "对**维度分**的贡献",而维度分与复合分已从 payload 整层删掉
                     (理由见 `research_mapper` 返回处那段注释)。逐槽的诚实呈现是
                     上面「行为指标观测明细」里那几张表:原始值 + 本场观测区间 +
                     有效样本量 + 置信度。
                     ⚠️ 标题留到了 2026-10-25 深夜才摘 —— 报告末尾一直挂着一个
                     **下面什么都没有**的二级标题,读者会以为内容丢了。
                     钉子:`tests/test_the_report_stops_saying_wrong_things.py`。 -->

                <footer style="text-align:center; margin-top:50px; color:#888;">
                    JingXin Multi-modal Assessment System | Auto-Generated Report
                </footer>
            </div>
        </body>
        </html>
        """
        return html


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口:`--session-id` 定要描述的那一场(M2:可以显式指定;不给就取最新一场)。

    **没产出报告就返回非 0**(第 17 条 / 复审 I2 的额外症状):总控面板
    `app.py:run_script` 只看子进程的 returncode —— "什么都没生成但 exit 0"
    会被显示成「任务完成!」。
    """
    import argparse

    ap = argparse.ArgumentParser(description="生成行为观测报告")
    ap.add_argument("--session-id", default=None,
                    help="要描述的那场会话 id;缺省取最新一场")
    args = ap.parse_args(argv)
    return 0 if ReportGenerator().generate_report(args.session_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())
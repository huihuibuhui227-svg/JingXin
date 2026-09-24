# report_frontend/report_generator.py

import os
import webbrowser
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from .data_loader import LogDataLoader
from .feature_engine import PsychologicalFeatureEngine
from .research_mapper import ResearchCapabilityMapper
from .visualizer import ReportVisualizer


def _interval_text(interval) -> str:
    """本场会话内观测区间。拿不到就说清是缺什么,而不是留空。

    区间是 `原始值 ± 会话内标准差`(与原始值同量纲、来自本场会话自己的测量),
    不是人群中位置 —— spec §5.5 禁止任何暗示人群位置的区间。
    """
    if not interval:
        return "未采集到会话内变异信息"
    return f"{interval[0]} – {interval[1]}"


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
        """从磁盘 CSV 文件生成评估报告（批处理模式）"""
        print("\n" + "=" * 70)
        print("🚀 启动 JingXin 面试行为观测报告生成系统 (批量模式)")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"Research_Assessment_Report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)

        try:
            loader = LogDataLoader()
            data = loader.get_fused_latest_data()
            if not data or 'face' not in data: raise ValueError("无面部数据")

            engine = PsychologicalFeatureEngine(data)
            features = engine.extract_all_features()

            mapper = ResearchCapabilityMapper()
            result = mapper.map_features_to_scores(features)

            viz = ReportVisualizer(output_dir=self.output_dir)
            chart_paths = viz.generate_all_charts(result, df_face=data['face'])
            static_images = self._scan_static_images()

            html_content = self._build_html_report(result, chart_paths, features, data, static_images)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            webbrowser.open('file://' + os.path.realpath(report_path))
            print(f"\n✅ 报告已生成并打开：{report_path}")
            return report_path

        except Exception as e:
            print(f"❌ 错误：{e}")
            return ""

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

            engine = PsychologicalFeatureEngine(data)
            features = engine.extract_all_features()

            mapper = ResearchCapabilityMapper()
            result = mapper.map_features_to_scores(features)

            viz = ReportVisualizer(output_dir=self.output_dir)
            chart_paths = viz.generate_all_charts(result, df_face=data.get('face'))
            static_images = self._scan_static_images()

            html_content = self._build_html_report(result, chart_paths, features, data, static_images)

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
        if dim["score"] is None:
            gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
            return f"""
            <h3>{dim['display_name']}</h3>
            <p><strong>证据不足</strong> —— 本次未采集到足以评估该行为线索的有效样本。</p>
            <ul>{gaps}</ul>
            """

        rows = "".join(
            f"<tr><td>{e['human_name']}</td><td>{e['raw_value']}</td>"
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

    def _build_html_report(self, result: Dict[str, Any], chart_paths: Dict[str, Any],
                           features: Dict[str, Any], data: Dict[str, pd.DataFrame],
                           static_images: List[str]) -> str:

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
            f"原始值 {slot['raw_value']}；"
            f"本场会话内观测区间 {_interval_text(slot['observed_interval'])}"
            f"（与原始值同量纲）；有效样本量 {slot['n_valid']}</li>"
            for slot in coverage["passed_slots"]
        )
        basis_html = (f"<ul>{basis_items}</ul>" if basis_items
                      else "<p>本次会话没有指标通过证据门。</p>")

        evidence_html_list = []
        for key, path in chart_paths.get('evidence', {}).items():
            dim_name = result['dimensions'][key]['display_name']
            ev_html = get_chart_iframe(path, "400")
            # 此处曾有「🧠 判推」框,渲染 result['dimensions'][key]['narrative']。
            # 该字段的产出方(template 填空式推断)已随 spec §5.4 一并删除,
            # 且报告本就不该给判推 —— 故整框移除,只留证据图。
            evidence_html_list.append(f"""
            <div class="card">
                <h3>🔍 {dim_name} - 证据链</h3>
                <div class="chart-container">{ev_html}</div>
            </div>
            """)

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

                <!-- 逐维观测明细:过门指标 + 未过门缺口 -->
                <div class="card">
                    <h2>📑 行为指标观测明细</h2>
                    {deep_analysis_html}
                </div>

                <div class="card">
                    <h3>📊 五维证据覆盖</h3>
                    <div class="chart-container">{get_chart_iframe(chart_paths.get('radar'), '500')}</div>
                </div>

                <h2>🔍 分维度证据链</h2>
                {''.join(evidence_html_list)}

                <footer style="text-align:center; margin-top:50px; color:#888;">
                    JingXin Multi-modal Assessment System | Auto-Generated Report
                </footer>
            </div>
        </body>
        </html>
        """
        return html


if __name__ == "__main__":
    ReportGenerator().generate_report()
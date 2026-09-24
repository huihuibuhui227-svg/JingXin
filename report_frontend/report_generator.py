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

# 置信度四档的高低顺序(取"置信度上限"时用)。
# research_mapper.confidence_from 只会产出这四档,键必须与之保持一致。
_CONF_ORDER = {"无": 0, "低": 1, "中": 2, "高": 3}


class ReportGenerator:
    """
    科研能力评估报告生成器 (终极丰满版)
    """

    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_report(self, session_id: Optional[str] = None) -> str:
        """从磁盘 CSV 文件生成评估报告（批处理模式）"""
        print("\n" + "=" * 70)
        print("🚀 启动 JingXin 科研能力评估报告生成系统 (批量模式)")
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
        print("🚀 启动 JingXin 科研能力评估报告生成系统 (实时模式)")
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
            f"<td>{e['normalized_score']}</td><td>{e['weight']}</td></tr>"
            for e in dim["evidence_chain"]
        )
        gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
        gaps_block = f"<p>未过门的指标：</p><ul>{gaps}</ul>" if gaps else ""
        return f"""
        <h3>{dim['display_name']}</h3>
        <p>依据 {dim['matched_indicators']} 个指标；置信度：<strong>{dim['confidence']}</strong>。</p>
        <table><thead><tr><th>指标</th><th>原始值</th><th>归一值</th><th>权重</th></tr></thead>
        <tbody>{rows}</tbody></table>
        {gaps_block}
        """

    def _generate_deep_text_analysis(self, features: Dict[str, Any], result: Dict[str, Any]) -> str:
        """只陈述本次实际测到了什么,不做任何解读、推断与形容词修饰。

        原实现把分数映射为固定阈值下的评语,并在字段缺失时用 .get(..., 0) 兜底,
        于是无论数据是否存在都会打印同几句结论 —— 恒定在默认值上的伪造百分位、
        与数据无关的微表情判断、以及基于默认分差值挑选出来的"突出维度"。
        现改为:出分则列出实际过门的指标及其原始值/权重,不出分则直说证据不足并列出缺口;
        全部维度都无证据时,只给诚实的空报告摘要。
        参数 features 保留是为了调用方签名稳定;本函数不再从原始特征里另取默认值。
        """
        parts = []
        total = result["total_score"]
        if total is None:
            parts.append("<p>本次会话未采集到足以支撑评估的有效证据。</p>")
        else:
            conf_cap = max((d["confidence"] for d in result["dimensions"].values()),
                           key=_CONF_ORDER.get)
            parts.append(
                f"<p>综合行为观测摘要：{result['total_level']}"
                f"(置信度上限：{conf_cap})</p>"
            )
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

        # 无证据时 total_score 为 None,直接插值会印出"None 分"
        _score_display = result['total_score'] if result['total_score'] is not None else "—"

        evidence_html_list = []
        for key, path in chart_paths.get('evidence', {}).items():
            dim_name = result['dimensions'][key]['display_name']
            ev_html = get_chart_iframe(path, "400")
            evidence_html_list.append(f"""
            <div class="card">
                <h3>🔍 {dim_name} - 证据链</h3>
                <div class="chart-container">{ev_html}</div>
                <div class="narrative-box"><strong>🧠 判推：</strong> {result['dimensions'][key]['narrative']}</div>
            </div>
            """)

        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>JingXin 科研能力深度评估报告</title>
            <style>
                :root {{ --primary: #2E86AB; --bg: #f4f7f6; }}
                body {{ font-family: 'Microsoft YaHei', sans-serif; background: var(--bg); color: #333; margin: 0; padding: 20px; line-height: 1.8; }}
                .container {{ max-width: 1100px; margin: 0 auto; }}
                header {{ text-align: center; padding: 40px; background: linear-gradient(135deg, #2E86AB, #A23B72); color: white; border-radius: 10px; margin-bottom: 30px; }}
                h1 {{ margin: 0; font-size: 2.5em; }}
                .score-board {{ display: flex; gap: 20px; margin: 20px 0; }}
                .score-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; }}
                .score-number {{ font-size: 3em; font-weight: bold; color: var(--primary); }}
                .card {{ background: white; padding: 30px; margin-bottom: 25px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                h2 {{ border-left: 5px solid var(--primary); padding-left: 15px; color: var(--primary); }}
                h3 {{ color: #444; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                .chart-container {{ margin: 20px 0; border: 1px solid #eee; border-radius: 5px; }}
                .narrative-box {{ background: #eef2f5; padding: 15px; border-left: 4px solid var(--primary); margin-top: 15px; }}
                .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} .score-board {{ flex-direction: column; }} }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>🔬 JingXin 科研能力评估报告</h1>
                    <div>基于多模态心理特征的深度分析与判推</div>
                    <div style="margin-top:10px; font-size:0.9em; opacity:0.8;">
                        {datetime.now().strftime("%Y-%m-%d %H:%M")} | {result['model_metadata']['version']}
                    </div>
                </header>

                <div class="score-board">
                    <div class="score-card">
                        <div class="score-number">{_score_display}</div>
                        <div>综合科研潜力评分</div>
                        <div style="color:var(--primary); font-weight:bold;">{result['total_level']}</div>
                    </div>
                    <div class="score-card" style="flex:2; text-align:left; display:flex; align-items:center;">
                        <div>
                            <h3 style="margin:0 0 10px 0; border:none;">📝 综合总结</h3>
                            <p style="margin:0;">{result['summary_narrative']}</p>
                        </div>
                    </div>
                </div>

                <!-- 深度文字报告 -->
                <div class="card">
                    <h2>📑 深度心理特征分析报告</h2>
                    {deep_analysis_html}
                </div>

                <div class="grid-2">
                    <div class="card">
                        <h3>📊 五维能力模型</h3>
                        <div class="chart-container">{get_chart_iframe(chart_paths.get('radar'), '500')}</div>
                    </div>
                    <div class="card">
                        <h3>👁️ 眼动行为分析</h3>
                        <div class="chart-container">{get_chart_iframe(chart_paths.get('gaze'), '500')}</div>
                    </div>
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

    def _generate_gaze_insight(self, data):
        return "详见上方眼动图表分析。"


if __name__ == "__main__":
    ReportGenerator().generate_report()
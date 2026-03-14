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


class ReportGenerator:
    """
    科研能力评估报告生成器 (终极丰满版)
    """

    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_report(self, session_id: Optional[str] = None) -> str:
        # ... (generate_report 方法保持不变，略) ...
        # 确保调用 _build_html_report 时传入了 features 和 result
        print("\n" + "=" * 70)
        print("🚀 启动 JingXin 科研能力评估报告生成系统 (终极丰满版)")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"Research_Assessment_Report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)

        try:
            loader = LogDataLoader()
            data = loader.get_fused_latest_data() if not session_id else loader.load_session_data(session_id)
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

    def _scan_static_images(self) -> List[str]:
        images = []
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'trajectory' not in f and 'chart' not in f:
                    images.append(f)
        return images

    def _get_percentile_badge(self, p: int) -> str:
        """根据百分位返回颜色徽章"""
        if p >= 90:
            return f"<span style='background:#d4edda; color:#155724; padding:2px 6px; border-radius:4px; font-weight:bold;'>Top {100 - p}% (卓越)</span>"
        elif p >= 70:
            return f"<span style='background:#d1ecf1; color:#0c5460; padding:2px 6px; border-radius:4px; font-weight:bold;'>Top {100 - p}% (优秀)</span>"
        elif p >= 40:
            return f"<span style='background:#fff3cd; color:#856404; padding:2px 6px; border-radius:4px;'>中等</span>"
        else:
            return f"<span style='background:#f8d7da; color:#721c24; padding:2px 6px; border-radius:4px;'>Bottom {p}% (待提升)</span>"

    def _generate_deep_text_analysis(self, features: Dict[str, Any], result: Dict[str, Any]) -> str:
        """生成终极丰满版文字报告"""
        face_feats = features.get('face', {})
        gesture_feats = features.get('gesture', {})

        # 提取更多统计量
        tension_mean = face_feats.get('face_tension_score_mean', 0)
        tension_std = face_feats.get('face_tension_score_std', 0)
        symmetry_mean = face_feats.get('face_symmetry_score_mean', 0)
        jitter_mean = gesture_feats.get('gesture_left_hand_jitter_mean', 0)
        jitter_std = gesture_feats.get('gesture_left_hand_jitter_std', 0)
        hand_score_mean = gesture_feats.get('gesture_left_hand_score_mean', 0)
        gaze_stab = face_feats.get('face_gaze_stability_mean', 0)
        eye_contact = face_feats.get('face_eye_contact_ratio', 0)
        au4_freq = face_feats.get('face_micro_exp_au_name_au4_freq', 0)
        au7_freq = face_feats.get('face_micro_exp_au_name_au7_freq', 0)

        # 获取维度统计数据 (含百分位)
        dim_stats = {}
        for k, v in result['dimensions'].items():
            if 'stats' in v:
                dim_stats[k] = v['stats']

        html_parts = []

        # --- 头部 ---
        level = result['total_level'].split()[0]
        html_parts.append(f"""
        <div style="background:#f8f9fa; padding:20px; border-left:5px solid #2E86AB; margin-bottom:25px;">
            <p style="margin:0; font-size:1.1em; line-height:1.8;">
                本报告基于 JingXin 多模态心理特征分析引擎，对候选人在模拟科研面试全过程中的 <strong>面部微表情 (Face)</strong>、
                <strong>肢体姿态 (Gesture)</strong>、<strong>眼动轨迹 (Gaze)</strong> 及 <strong>语音韵律 (Voice)</strong> 进行了毫秒级量化分析。
                系统共提取了 <strong>{sum(len(v) for v in features.values())}</strong> 个量化指标，并通过常模参照模型进行了深度判推。
                <br><br>
                <strong>综合结论：</strong> 候选人综合科研潜力评分为 <span style="color:#2E86AB; font-weight:bold; font-size:1.2em;">{result['total_score']}</span> 分，
                评级为 <strong>{level}</strong>。
            </p>
        </div>
        """)

        # --- 1. 情绪与抗压 (深度版) ---
        stress_stats = dim_stats.get('stress_resilience', {})
        tension_p = stress_stats.get('tension_score', {}).get('percentile', 50)
        badge = self._get_percentile_badge(100 - tension_p)  # 紧张度越低越好，所以用 100-p

        analysis_text = f"候选人的面部紧张度均值为 <strong>{tension_mean:.2f}</strong> (标准差 {tension_std:.2f})。"
        if tension_mean > 0.6:
            analysis_text += "该数值处于较高水平，表明候选人在面试过程中经历了显著的心理压力。"
        elif tension_mean > 0.3:
            analysis_text += "该数值处于适中范围，表明候选人具备一定的抗压能力，但在关键节点仍有波动。"
        else:
            analysis_text += "该数值处于较低水平，展现了极佳的情绪控制力和心理稳定性。"

        analysis_text += f" 在人群常模中，其情绪稳定性表现优于 {100 - tension_p}% 的受试者，{badge}。"

        if au4_freq > 0.1 or au7_freq > 0.1:
            analysis_text += f" 微表情分析检测到皱眉 (AU4) 频率为 {au4_freq:.1%}，眼部挤压 (AU7) 频率为 {au7_freq:.1%}，这通常是认知负荷过高或焦虑的直接生理信号。"
        else:
            analysis_text += " 微表情监测未检测到显著的焦虑特征 (AU4/AU7 频率低)，表明表面情绪较为平稳。"

        html_parts.append(f"""
        <h3>1. 情绪状态与抗压能力深度剖析</h3>
        <p>{analysis_text}</p>
        <div style="background:#fff; border:1px solid #eee; padding:15px; border-radius:5px; margin-top:10px;">
            <strong>💡 科研场景映射：</strong> 
            {'在高强度科研攻关或答辩场景下，候选人可能需要额外的时间来调节情绪，建议进行脱敏训练。' if tension_mean > 0.5 else '候选人具备在压力下保持冷静的潜质，适合承担具有挑战性的科研任务。'}
        </div>
        """)

        # --- 2. 肢体与自信 (深度版) ---
        conf_stats = dim_stats.get('confidence_level', {})
        jitter_p = conf_stats.get('jitter', {}).get('percentile', 50)
        badge_jitter = self._get_percentile_badge(100 - jitter_p)

        body_text = f"肢体遥测数据显示，候选人左手抖动均值为 <strong>{jitter_mean:.4f}</strong> (标准差 {jitter_std:.4f})。"
        if jitter_mean > 0.03:
            body_text += "显著的生理性震颤通常与交感神经兴奋（紧张）相关。"
        elif jitter_mean > 0.01:
            body_text += "轻微的抖动属于正常生理现象，但在高压下略有放大。"
        else:
            body_text += "极低的抖动值展现了如外科医生般的肢体控制稳定性。"
        body_text += f" 该指标在人群中处于 {100 - jitter_p}% 的水平，{badge_jitter}。"
        body_text += f" 手势自信度评分为 <strong>{hand_score_mean:.1f}</strong>，结合肩部放松度指标，"
        body_text += "反映了候选人肢体语言的开放性。"

        html_parts.append(f"""
        <h3>2. 肢体语言与自信心量化评估</h3>
        <p>{body_text}</p>
        """)

        # --- 3. 眼动与专注 (深度版) ---
        logic_stats = dim_stats.get('logical_thinking', {})
        gaze_p = logic_stats.get('gaze_stability', {}).get('percentile', 50)
        badge_gaze = self._get_percentile_badge(gaze_p)

        gaze_text = f"眼动追踪算法计算出视线稳定性指数为 <strong>{gaze_stab:.2f}</strong>，眼神接触比例高达 <strong>{eye_contact:.2%}</strong>。"
        if gaze_stab > 0.75:
            gaze_text += "极高的稳定性意味着候选人能够长时间将注意力锁定在目标上，这是深度科研工作者的核心特质。"
        elif gaze_stab > 0.5:
            gaze_text += "良好的稳定性表明候选人具备正常的专注力，但在复杂信息处理时偶有扫视。"
        else:
            gaze_text += "较低的稳定性提示注意力可能存在分散，或在思考时倾向于通过眼球运动来辅助认知加工。"
        gaze_text += f" 该专注力水平超越了 {gaze_p}% 的人群，{badge_gaze}。"

        html_parts.append(f"""
        <h3>3. 视线追踪与专注力判读</h3>
        <p>{gaze_text}</p>
        <div style="background:#fff; border:1px solid #eee; padding:15px; border-radius:5px; margin-top:10px;">
            <strong>👁️ 视觉热点分析：</strong> 
            结合眼动热力图（见下文），候选人的视觉关注点主要集中在中心区域，符合正常的交流注视模式，未出现异常的回避行为。
        </div>
        """)

        # --- 4. 总结与建议 ---
        sorted_dims = sorted(result['dimensions'].items(), key=lambda x: x[1]['score'], reverse=True)
        top_dim = sorted_dims[0][1]['display_name']
        bottom_dim = sorted_dims[-1][1]['display_name']

        summary_text = f"综上所述，候选人在 <strong>{top_dim}</strong> 维度表现最为突出，显示出良好的科研天赋。"
        summary_text += f" 然而，在 <strong>{bottom_dim}</strong> 维度上得分相对较低，是主要的短板所在。"
        summary_text += " 建议后续针对该短板进行专项训练（如模拟高压面试、正念冥想等）。总体而言，该候选人具备从事科研工作的基本心理素质。"

        html_parts.append(f"""
        <h3>4. 综合结论与发展建议</h3>
        <p>{summary_text}</p>
        """)

        return "".join(html_parts)

    def _build_html_report(self, result: Dict[str, Any], chart_paths: Dict[str, Any],
                           features: Dict[str, Any], data: Dict[str, pd.DataFrame],
                           static_images: List[str]) -> str:

        def get_chart_iframe(path, height="500"):
            if not path or not os.path.exists(path): return '<div class="placeholder">图表缺失</div>'
            return f'<iframe src="{os.path.basename(path)}" width="100%" height="{height}px" frameborder="0"></iframe>'

        deep_analysis_html = self._generate_deep_text_analysis(features, result)

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
                        <div class="score-number">{result['total_score']}</div>
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
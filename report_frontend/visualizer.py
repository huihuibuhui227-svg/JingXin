# report_frontend/visualizer.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import os
import time

warnings.filterwarnings('ignore')

# 【关键】设置 Plotly 默认字体以支持中文
PLOTLY_FONT_FAMILY = "Microsoft YaHei, PingFang SC, SimHei, sans-serif"


class ReportVisualizer:
    """
    行为观测报告可视化引擎 (雷达图 + 逐维证据图 + 自动保存)

    【说明】
    1. 眼动图:M3 之前不生成 —— 现有坐标撑不起"注视"这个构念,见
       create_gaze_plot_from_df 的说明(spec §5.5)。
    2. 自动保存机制:所有图表自动保存至 data/output 目录。
    3. 路径管理:返回图表文件路径,方便 HTML 报告引用。
    """

    def __init__(self, output_dir: str = "data/output"):
        self.font_family = PLOTLY_FONT_FAMILY
        self.colors = {
            "primary": "#2E86AB", "secondary": "#A23B72", "success": "#28A745",
            "warning": "#FFC107", "danger": "#DC3545", "info": "#17A2B8",
            "baseline": "#6c757d", "grid": "#e9ecef", "heatmap": "Viridis"
        }

        # 【核心】初始化输出目录
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📂 已创建输出目录：{self.output_dir}")

        # 记录已生成的文件路径
        self.generated_files = []

    def _save_fig(self, fig: go.Figure, filename: str) -> str:
        """
        保存图表为独立 HTML 文件，并返回相对路径
        """
        timestamp = int(time.time())
        safe_filename = f"{filename}_{timestamp}.html"
        full_path = os.path.join(self.output_dir, safe_filename)

        # 保存为完整 HTML 文件 (包含 JS)
        fig.write_html(full_path, include_plotlyjs=True, full_html=True, default_height='500px')

        # 返回相对于项目根目录的路径 (假设当前脚本在项目根目录或子目录运行)
        # 为了通用性，我们返回绝对路径，report_generator 可以处理，或者计算相对路径
        # 这里简单返回绝对路径，确保一定能找到
        self.generated_files.append(full_path)
        return full_path

    def _build_radar_figure(self, result: Dict[str, Any]) -> go.Figure:
        """构造证据覆盖雷达图,不落盘 —— 便于测试(spec §6)。

        ⚠️ 半径**不再是维度得分**。0–100 的维度分是未标定标尺上的复合点分,spec §5.4
        :157-158 / §5.6 只允许它以「区间 + 置信度 + 依据」出现,而一条雷达轴既不是区间
        也不是依据 —— 画出来等于把被停用的点分又渲染一遍(实测 1/20 覆盖下会画出
        一根指到 100 的轴,读起来就是"这一维满分")。
        改画**证据覆盖**:每个维度有几个指标槽通过了证据门(0 到该维槽数)。
        """
        dimensions = result['dimensions']
        categories = []
        passed_counts = []
        slot_counts = []

        dim_order = ['logical_thinking', 'stress_resilience', 'communication_fluency', 'confidence_level',
                     'cognitive_efficiency']

        for key in dim_order:
            if key in dimensions:
                dim = dimensions[key]
                cat_name = dim['display_name'].replace("与", "&").replace("度", "")
                got, _, total = dim['matched_indicators'].partition("/")
                categories.append(cat_name)
                passed_counts.append(int(got))
                slot_counts.append(int(total))
            else:
                categories.append("未知")
                passed_counts.append(0)
                slot_counts.append(0)

        # 闭合多边形(与原先的候选分多边形同一处理)
        categories += [categories[0]]
        passed_counts += [passed_counts[0]]

        fig = go.Figure()
        # 说明:曾有的 '常模基准' 虚线来自硬编码的 [60]*5,并无真实常模出处,
        # 却以权威对比的形式呈现,故一并删除(spec §5.5:没有真实常模就不画常模线)。
        fig.add_trace(go.Scatterpolar(r=passed_counts, theta=categories, fill='toself',
                                      name='通过证据门的指标槽数',
                                      line_color=self.colors['primary'], fillcolor='rgba(46, 134, 171, 0.4)'))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(slot_counts) or 1],
                                       tickfont=dict(family=self.font_family)),
                       angularaxis=dict(tickfont=dict(family=self.font_family), rotation=90, direction='clockwise')),
            title=dict(text="📊 五维证据覆盖（通过证据门的指标槽数）", x=0.5,
                       font=dict(family=self.font_family, size=18)),
            height=500, showlegend=True
        )

        return fig

    def create_capability_radar(self, result: Dict[str, Any]) -> str:
        """创建雷达图并保存"""
        fig = self._build_radar_figure(result)
        return self._save_fig(fig, "radar_chart")

    def create_evidence_bar_chart(self, result: Dict[str, Any], dimension_key: str) -> str:
        """创建证据链条形图并保存"""
        if dimension_key not in result['dimensions']:
            return ""

        data = result['dimensions'][dimension_key]
        evidence = data['evidence_chain']

        labels, contributions, colors, hover_texts = [], [], [], []

        for item in evidence:
            if item['raw_value'] is None: continue
            labels.append(item['human_name'])
            contrib = item['contribution']
            contributions.append(contrib)

            if contrib > 0:
                colors.append(self.colors['success']); status = "正向"
            elif contrib < 0:
                colors.append(self.colors['danger']); status = "负向"
            else:
                colors.append(self.colors['info']); status = "中性"

            hover_texts.append(
                f"<b>{item['human_name']}</b><br>贡献：{contrib:.3f} ({status})<br>原始值：{item['raw_value']}<extra></extra>")

        fig = go.Figure(go.Bar(y=labels, x=contributions, orientation='h', marker_color=colors, hovertext=hover_texts,
                               hoverinfo='text'))
        fig.update_layout(
            title=dict(text=f"🔍 [{data['display_name']}] 证据链贡献度", font=dict(family=self.font_family, size=16)),
            xaxis_title="贡献值", yaxis_title="指标",
            yaxis=dict(tickfont=dict(family=self.font_family)),
            height=max(300, len(labels) * 40), margin=dict(l=150, r=20, t=60, b=20)
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        return self._save_fig(fig, f"evidence_{dimension_key}")

    def create_gaze_plot_from_df(self, df_face: pd.DataFrame) -> Optional[str]:
        """M3 之前不生成眼动图。

        gaze_direction_y 是解剖常量、iris_x/y 是图像归一化坐标(编码人脸位置),
        两者都不能支撑"注视热力图"这个标题。见 spec §5.5。

        原实现在此画两张图:左图是 gaze_direction_x/y 的二维直方图(其中 y 恒负),
        右图是左右眼 iris 的连线(实为"人脸在画面里怎么动"),另有一处
        add_shape(rect, x0=-1, y0=-1, x1=1, y1=1) —— 而 iris 坐标归一化在 [0,1],
        画 [-1,1] 的框没有语义。三处一并删除,待 M3 换成眼内相对坐标后再实现。

        参数与返回值保持不变:调用方(generate_all_charts)无需改动,M3 可直接续写。
        """
        return None

    def generate_all_charts(self, result: Dict[str, Any], df_face: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        生成所有图表并保存
        :param result: Mapper 的结果
        :param df_face: 原始面部数据 DataFrame (用于绘制眼动图)
        :return: 包含所有文件路径的字典
        """
        charts = {}

        # 1. 雷达图
        charts['radar'] = self.create_capability_radar(result)
        print(f"   📈 已保存：雷达图 -> {charts['radar']}")

        # 2. 证据链图 (每个维度一个)
        charts['evidence'] = {}
        for key in result['dimensions'].keys():
            path = self.create_evidence_bar_chart(result, key)
            if path:
                charts['evidence'][key] = path
                print(f"   📈 已保存：[{key}] 证据图 -> {path}")

        # 3. 眼动图 —— M3 前恒为 None(spec §5.5),故不再区分"有/无数据"两种分支
        charts['gaze'] = self.create_gaze_plot_from_df(df_face)
        print("   ℹ️ 眼动图:M3 前不生成(现有坐标不能支撑'注视'构念,spec §5.5)。")

        return charts


# --- 本地测试入口 ---
if __name__ == "__main__":
    from data_loader import LogDataLoader
    from feature_engine import PsychologicalFeatureEngine
    from research_mapper import ResearchCapabilityMapper

    print("=== 测试 Visualizer (雷达图 + 证据图 + 自动保存) ===")

    # 1. 准备数据
    loader = LogDataLoader()
    data = loader.get_fused_latest_data()

    if data and 'face' in data :
        engine = PsychologicalFeatureEngine(data)
        features = engine.extract_all_features()

        mapper = ResearchCapabilityMapper()
        result = mapper.map_features_to_scores(features)

        # 2. 初始化 Visualizer (自动创建 data/output)
        viz = ReportVisualizer(output_dir="data/output")

        # 3. 生成所有图表 (传入原始 df_face)
        chart_paths = viz.generate_all_charts(result, df_face=data['face'])

        print("\n✅ 所有图表已生成并保存至 data/output 目录！")
        print(f"📂 文件列表：{list(chart_paths.values())}")
        print("💡 请前往 D:/jingxin/data/output 查看生成的 HTML 文件。")
    else:
        print("❌ 数据加载失败，无法测试可视化。")
# report_frontend/visualizer.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    科研能力评估报告可视化引擎 (终极版：含眼动轨迹 + 自动保存)

    【功能升级】
    1. 新增眼动热力轨迹图：直观展示视线聚焦区域与扫描路径。
    2. 自动保存机制：所有图表自动保存至 data/output 目录。
    3. 路径管理：返回相对路径，方便 HTML 报告引用。
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

    def create_capability_radar(self, result: Dict[str, Any]) -> str:
        """创建雷达图并保存"""
        dimensions = result['dimensions']
        categories = []
        scores = []
        baselines = [60, 60, 60, 60, 60]

        dim_order = ['logical_thinking', 'stress_resilience', 'communication_fluency', 'confidence_level',
                     'cognitive_efficiency']

        for key in dim_order:
            if key in dimensions:
                cat_name = dimensions[key]['display_name'].replace("与", "&").replace("度", "")
                categories.append(cat_name)
                scores.append(dimensions[key]['score'])
            else:
                categories.append("未知")
                scores.append(0)

        categories += [categories[0]]
        scores += [scores[0]]
        baselines += [baselines[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=scores, theta=categories, fill='toself', name='候选人得分',
                                      line_color=self.colors['primary'], fillcolor='rgba(46, 134, 171, 0.4)'))
        fig.add_trace(go.Scatterpolar(r=baselines, theta=categories, fill='none', name='常模基准',
                                      line_color=self.colors['baseline'], line_dash='dot'))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(family=self.font_family)),
                       angularaxis=dict(tickfont=dict(family=self.font_family), rotation=90, direction='clockwise')),
            title=dict(text="📊 科研能力五维模型评估", x=0.5, font=dict(family=self.font_family, size=18)),
            height=500, showlegend=True
        )

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

    def create_gaze_trajectory_heatmap(self, features: Dict[str, Any]) -> Optional[str]:
        """
        【新增】基于眼动坐标绘制视线轨迹与热力图
        输入：features 字典 (需包含 face_gaze_direction_x/y_mean 等，或者最好有原始 DataFrame)
        *注意*：由于 features 是统计值，这里我们模拟或尝试从 features 中寻找序列数据。
        如果 feature_engine 没有输出序列，我们需要从 data_loader 传入原始 df。

        *修正策略*：为了演示效果，如果传入的是统计值，我们绘制一个示意性的“关注区域图”。
        如果后续能传入原始 df，则绘制真实轨迹。
        此处假设我们无法直接访问原始 df，我们将利用统计值绘制一个“虚拟分布”或提示需要原始数据。

        *最佳实践*：修改函数签名，允许传入原始 data_frames。
        """
        # 这里为了完整性，我们假设 visualizer 可以接收原始数据 frames
        # 但在当前架构下，我们主要依赖 features。
        # 如果 features 里没有序列，我们创建一个基于统计值的“注意力焦点图”

        # 尝试提取 gaze 相关的统计值来推断焦点
        # 如果没有具体序列，我们画一个标准的“中心聚焦”示意图作为占位，或者跳过
        # *真正有用的实现*：需要修改 generate_all_charts 接收 data_frames

        # 既然你要求基于“眼动坐标数据”，我假设我们可以访问到原始数据或者 feature_engine 输出了足够的信息。
        # 但目前的 feature_engine 只输出了 mean/std。
        # **解决方案**：我们在 generate_all_charts 中传入原始 data，在这里使用。
        # 为了保持接口简洁，我先写一个接收 data_frames 的版本，并在下方调用处说明。
        return None

    def create_gaze_plot_from_df(self, df_face: pd.DataFrame) -> Optional[str]:
        """
        真正的眼动绘图函数：接收原始 Face DataFrame
        绘制：1. 视线散点热力图 2. 左右眼 iris 移动轨迹
        """
        if df_face is None or df_face.empty:
            return None

        # 检查是否有眼动列
        has_left = 'left_iris_x' in df_face.columns and 'left_iris_y' in df_face.columns
        has_right = 'right_iris_x' in df_face.columns and 'right_iris_y' in df_face.columns
        has_gaze = 'gaze_direction_x' in df_face.columns and 'gaze_direction_y' in df_face.columns

        if not (has_left or has_right or has_gaze):
            return None

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("👁️ 视线注视热力图 (Gaze Heatmap)", "👀 虹膜运动轨迹 (Iris Trajectory)"))

        # 1. 热力图 (使用 gaze_direction 或 平均 iris 位置)
        if has_gaze:
            x_col, y_col = 'gaze_direction_x', 'gaze_direction_y'
            # 注意：gaze_direction 可能是相对值，这里直接画散点密度
            fig.add_trace(go.Histogram2dContour(
                x=df_face[x_col], y=df_face[y_col],
                colorscale='Viridis', showscale=True,
                hoverinfo='skip'
            ), row=1, col=1)

            # 添加中心参考点 (假设 0,0 或 0.5,0.5 是中心，根据数据分布调整)
            # 这里添加一个矩形框表示屏幕/摄像头范围
            fig.add_shape(type="rect", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="Red", width=2), row=1, col=1)

        # 2. 轨迹图 (左右眼)
        if has_left:
            fig.add_trace(go.Scatter(
                x=df_face['left_iris_x'], y=df_face['left_iris_y'],
                mode='lines+markers', name='左眼 (Left)',
                line=dict(color='Blue', width=1), marker=dict(size=4),
                opacity=0.7
            ), row=1, col=2)

        if has_right:
            fig.add_trace(go.Scatter(
                x=df_face['right_iris_x'], y=df_face['right_iris_y'],
                mode='lines+markers', name='右眼 (Right)',
                line=dict(color='Red', width=1), marker=dict(size=4),
                opacity=0.7
            ), row=1, col=2)

        # 统一布局
        fig.update_layout(
            title=dict(text="🧠 眼动行为深度分析 (视线聚焦与运动轨迹)", x=0.5,
                       font=dict(family=self.font_family, size=18)),
            height=500, showlegend=True,
            font=dict(family=self.font_family)
        )

        fig.update_xaxes(title_text="X 坐标", row=1, col=1)
        fig.update_yaxes(title_text="Y 坐标", row=1, col=1)
        fig.update_xaxes(title_text="X 坐标", row=1, col=2)
        fig.update_yaxes(title_text="Y 坐标", row=1, col=2)

        return self._save_fig(fig, "gaze_trajectory")

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

        # 3. 【新增】眼动轨迹图
        if df_face is not None:
            gaze_path = self.create_gaze_plot_from_df(df_face)
            if gaze_path:
                charts['gaze'] = gaze_path
                print(f"   👁️ 已保存：眼动轨迹图 -> {gaze_path}")
            else:
                charts['gaze'] = None
                print("   ⚠️ 未找到眼动数据，跳过眼动图生成。")
        else:
            charts['gaze'] = None
            print("   ⚠️ 未传入面部原始数据，跳过眼动图生成。")

        return charts


# --- 本地测试入口 ---
if __name__ == "__main__":
    from data_loader import LogDataLoader
    from feature_engine import PsychologicalFeatureEngine
    from research_mapper import ResearchCapabilityMapper

    print("=== 测试 Visualizer (含眼动 + 自动保存) ===")

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
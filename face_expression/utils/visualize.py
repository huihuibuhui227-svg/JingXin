# face_expression/utils/visualize.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any

def _safe_json_loads(s: str) -> Dict[str, Any]:
    """安全解析 JSON 字符串（兼容单引号）"""
    if pd.isna(s) or s == '{}' or s == '':
        return {}
    try:
        return json.loads(s.replace("'", '"'))
    except:
        return {}

def plot_emotion_radar(emotion_vector: Dict[str, float], save_path: Optional[str] = None):
    """
    绘制情绪雷达图（支持多维情绪）
    """
    # 过滤掉 neutral/polite_smile 等非基本情绪（可配置）
    basic_emotions = ["happy", "sadness", "anger", "fear", "surprise", "disgust"]
    values = [emotion_vector.get(e, 0) for e in basic_emotions]

    angles = np.linspace(0, 2 * np.pi, len(basic_emotions), endpoint=False).tolist()
    values += values[:1]  # 闭合
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.6)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(basic_emotions, fontsize=12)
    ax.set_title("Emotion Radar Chart", fontsize=14, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_features_from_csv(csv_path: str = "face_au_log.csv") -> bool:
    """
    增强型可视化：支持多维情绪、心理信号、微表情、AU 时序
    """
    if not os.path.exists(csv_path):
        print(f"❌ 文件 {csv_path} 不存在，请先运行分析器")
        return False

    # 安全导入
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        return False

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("❌ CSV 为空")
            return False

        # === 解析嵌套字段 ===
        df['psychological_signals'] = df['psychological_signals'].apply(_safe_json_loads)
        df['micro_expressions'] = df['micro_expressions'].apply(_safe_json_loads)
        df['emotion_vector'] = df['emotion_vector'].apply(_safe_json_loads)

        # 提取紧张度
        df['tension_score'] = df['psychological_signals'].apply(lambda x: x.get('tension_score', 0))
        df['tension_level'] = df['psychological_signals'].apply(lambda x: x.get('tension_level', 'low'))

        # 提取主导情绪
        df['dominant_emotion'] = df['emotion_vector'].apply(lambda x: x.get('dominant_emotion', 'neutral'))
        df['confidence'] = df['emotion_vector'].apply(lambda x: x.get('confidence', 0))

        # 微表情时间点
        micro_times = df[df['micro_expressions'].astype(str) != '{}'].index.tolist()

        # 时间轴（使用 timestamp 或 index）
        if 'timestamp' in df.columns and df['timestamp'].dtype in ['float64', 'int64']:
            time_sec = (df['timestamp'] - df['timestamp'].iloc[0])
        else:
            time_sec = df.index / 30.0  # 默认 30fps

        # === 设置中文字体 ===
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        use_chinese = False
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                fig_test, ax_test = plt.subplots()
                ax_test.text(0.5, 0.5, '测试')
                plt.close(fig_test)
                use_chinese = True
                break
            except:
                continue

        title_prefix = "面部心理状态综合分析" if use_chinese else "Comprehensive Facial Psychological Analysis"

        # === 创建主图（4x2）===
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(title_prefix, fontsize=16)

        # 1. 紧张度 + 微表情标记
        axes[0, 0].plot(time_sec, df['tension_score'], 'r-', label='Tension Score')
        for t in micro_times:
            axes[0, 0].axvline(x=time_sec.iloc[t], color='purple', linestyle='--', alpha=0.7, label='Micro-exp' if t == micro_times[0] else "")
        axes[0, 0].set_ylabel('Tension (0-1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. 情绪置信度
        axes[0, 1].plot(time_sec, df['confidence'], 'b-', label='Emotion Confidence')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].grid(True)

        # 3. 基础 AU
        axes[1, 0].plot(time_sec, df.get('au4_frown', 0), label='AU4 (Frown)')
        axes[1, 0].plot(time_sec, df.get('au12_smile', 0), label='AU12 (Smile)')
        axes[1, 0].set_ylabel('AU Intensity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. 新增 AU
        axes[1, 1].plot(time_sec, df.get('au6_cheek_raise', 0), label='AU6 (Cheek)')
        axes[1, 1].plot(time_sec, df.get('au23_lip_compression', 0), label='AU23 (Lip Comp)')
        axes[1, 1].set_ylabel('New AU')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 5. 生理指标
        axes[2, 0].plot(time_sec, df.get('blink_rate_per_min', 0), 'k-', label='Blink Rate')
        axes[2, 0].set_ylabel('Blinks/min')
        axes[2, 0].grid(True)

        # 6. 对称性 & 专注度
        axes[2, 1].plot(time_sec, df.get('symmetry_score', 1), label='Symmetry')
        axes[2, 1].plot(time_sec, df.get('focus_score', 0.5), label='Focus')
        axes[2, 1].set_ylabel('Score')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # 7. 主导情绪（分类图）
        emotions = df['dominant_emotion'].astype('category')
        colors = plt.cm.tab10(np.linspace(0, 1, len(emotions.cat.categories)))
        emotion_colors = [colors[emotions.cat.categories.get_loc(e)] for e in emotions]
        axes[3, 0].scatter(time_sec, [0.5] * len(time_sec), c=emotion_colors, s=20, alpha=0.7)
        axes[3, 0].set_yticks([])
        axes[3, 0].set_ylabel('Dominant Emotion')
        axes[3, 0].grid(True)

        # 8. 情绪雷达图（最后一帧）
        if not df['emotion_vector'].empty:
            last_emotion = df['emotion_vector'].iloc[-1]
            if isinstance(last_emotion, dict) and last_emotion:
                basic_emotions = ["happy", "sadness", "anger", "fear", "surprise", "disgust"]
                values = [last_emotion.get(e, 0) for e in basic_emotions]
                angles = np.linspace(0, 2 * np.pi, len(basic_emotions), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                axes[3, 1].plot(angles, values, 'o-', color='blue')
                axes[3, 1].fill(angles, values, alpha=0.25, color='lightblue')
                axes[3, 1].set_xticks(angles[:-1])
                axes[3, 1].set_xticklabels(basic_emotions, rotation=30)
                axes[3, 1].set_ylim(0, 1)
                axes[3, 1].set_title('Last Frame Emotion Radar')

        plt.tight_layout()
        save_img_path = csv_path.replace('.csv', '_enhanced_analysis.png')
        plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
        plt.show()

        # === 单独保存雷达图 ===
        radar_path = csv_path.replace('.csv', '_emotion_radar.png')
        if last_emotion:
            plot_emotion_radar(last_emotion, radar_path)

        print(f"✅ 可视化已保存至:\n - {save_img_path}\n - {radar_path}")
        return True

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = plot_features_from_csv()
    if success:
        print("✅ 图表已成功显示并保存")
    else:
        print("❌ 图表生成失败")
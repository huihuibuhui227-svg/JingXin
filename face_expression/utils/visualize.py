import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

# 处理导入路径，支持直接运行和模块导入
try:
    from ..config import FACE_EXPRESSION_OUTPUT_DIR
except ImportError:
    # 直接运行时，从当前路径获取配置
    import sys
    config_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(config_dir))
    try:
        from config import FACE_EXPRESSION_OUTPUT_DIR
    except ImportError as e:
        # 如果还是失败，手动设置默认路径
        print(f"⚠️ 无法导入配置文件: {e}")
        # 获取项目根目录
        project_root = config_dir.parent
        FACE_EXPRESSION_OUTPUT_DIR = project_root / 'data' / 'output' / 'face_expression'
        FACE_EXPRESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 新增辅助函数：获取项目根目录 ===
def get_project_root() -> Path:
    """获取项目根目录（包含 jingxin/ 的父目录）"""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if parent.name == 'jingxin':
            return parent
    raise FileNotFoundError("❌ 未找到项目根目录 'jingxin/'")

# === 新增辅助函数：获取最新日志文件 ===
def get_latest_log_file(log_dir: str = "data/logs", prefix: str = "face_au_log") -> str:
    """获取指定目录下最新生成的 CSV 日志文件"""
    project_root = get_project_root()
    log_path = project_root / log_dir

    if not log_path.exists():
        print(f"❌ 日志目录 {log_path} 不存在")
        return ""

    csv_files = list(log_path.glob(f"{prefix}_*.csv"))
    if not csv_files:
        print(f"❌ 未找到以 {prefix}_ 开头的日志文件")
        return ""

    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    print(f"✅ 找到最新日志文件: {latest_file.name}")
    return str(latest_file)

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
    basic_emotions = ["happy", "sadness", "anger", "fear", "surprise", "disgust"]
    values = [emotion_vector.get(e, 0) for e in basic_emotions]

    angles = np.linspace(0, 2 * np.pi, len(basic_emotions), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.6)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(basic_emotions, fontsize=12)
    ax.set_title("情绪雷达图", fontsize=14, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_features_from_csv(csv_path: str = "") -> bool:
    """
    增强型可视化：支持多维情绪、心理信号、微表情、AU 时序、视线追踪
    """
    if not csv_path:
        csv_path = get_latest_log_file()
        if not csv_path:
            return False

    if not os.path.exists(csv_path):
        print(f"❌ 文件 {csv_path} 不存在，请先运行分析器")
        return False

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

        # 解析嵌套字段
        df['psychological_signals'] = df['psychological_signals'].apply(_safe_json_loads)
        df['micro_expressions'] = df['micro_expressions'].apply(_safe_json_loads)
        df['emotion_vector'] = df['emotion_vector'].apply(_safe_json_loads)

        # 提取紧张度
        df['tension_score'] = df['psychological_signals'].apply(lambda x: x.get('tension_score', 0))
        df['tension_level'] = df['psychological_signals'].apply(lambda x: x.get('tension_level', 'low'))

        # 提取主导情绪
        df['dominant_emotion'] = df['emotion_vector'].apply(lambda x: x.get('dominant_emotion', 'unknown'))
        df['confidence'] = df['emotion_vector'].apply(lambda x: x.get('confidence', 0))

        # 微表情时间点
        micro_times = df[df['micro_expressions'].astype(str) != '{}'].index.tolist()

        # 时间轴
        if 'timestamp' in df.columns and df['timestamp'].dtype in ['float64', 'int64']:
            time_sec = (df['timestamp'] - df['timestamp'].iloc[0])
        else:
            time_sec = df.index / 30.0

        # 设置中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei']
        use_chinese = False
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                use_chinese = True
                break
            except:
                continue

        title_prefix = "面部心理状态综合分析"

        # 创建 4x2 图表
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(title_prefix, fontsize=16)

        # 1. 紧张度 + 微表情标记
        axes[0, 0].plot(time_sec, df['tension_score'], 'r-', label='紧张度')
        for t in micro_times:
            axes[0, 0].axvline(x=time_sec.iloc[t], color='purple', linestyle='--', alpha=0.7, label='微表情' if t == micro_times[0] else "")
        axes[0, 0].set_ylabel('紧张度 (0-1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. 情绪置信度
        axes[0, 1].plot(time_sec, df['confidence'], 'b-', label='情绪置信度')
        axes[0, 1].set_ylabel('置信度')
        axes[0, 1].grid(True)

        # 3. 基础 AU
        axes[1, 0].plot(time_sec, df.get('au4_frown', 0), label='AU4 (皱眉)')
        axes[1, 0].plot(time_sec, df.get('au12_smile', 0), label='AU12 (微笑)')
        axes[1, 0].set_ylabel('AU 强度')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. 新增 AU
        axes[1, 1].plot(time_sec, df.get('au6_cheek_raise', 0), label='AU6 (脸颊抬起)')
        axes[1, 1].plot(time_sec, df.get('au23_lip_compression', 0), label='AU23 (嘴唇压缩)')
        axes[1, 1].set_ylabel('新 AU')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 5. 生理指标
        axes[2, 0].plot(time_sec, df.get('blink_rate_per_min', 0), 'k-', label='眨眼频率')
        axes[2, 0].set_ylabel('眨眼/分钟')
        axes[2, 0].grid(True)

        # 6. ✅ 视线方向图（新增）
        axes[2, 1].scatter(time_sec, df.get('gaze_direction_x', 0), label='视线 X', color='red', s=10)
        axes[2, 1].scatter(time_sec, df.get('gaze_direction_y', 0), label='视线 Y', color='blue', s=10)
        axes[2, 1].set_ylabel('视线方向')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # 7. 主导情绪（分类图）
        emotions = df['dominant_emotion'].astype('category')
        colors = plt.cm.tab10(np.linspace(0, 1, len(emotions.cat.categories)))
        emotion_colors = [colors[emotions.cat.categories.get_loc(e)] for e in emotions]
        axes[3, 0].scatter(time_sec, [0.5] * len(time_sec), c=emotion_colors, s=20, alpha=0.7)
        axes[3, 0].set_yticks([])
        axes[3, 0].set_ylabel('主导情绪')
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
                axes[3, 1].set_title('末帧情绪雷达图')

        plt.tight_layout()
        # 保存到output/face_expression目录
        log_filename = Path(csv_path).stem
        save_img_path = os.path.join(FACE_EXPRESSION_OUTPUT_DIR, f'{log_filename}_enhanced_analysis.png')
        plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
        plt.show()

        # 单独保存雷达图到output/face_expression目录
        radar_path = os.path.join(FACE_EXPRESSION_OUTPUT_DIR, f'{log_filename}_emotion_radar.png')
        if not df['emotion_vector'].empty:
            last_emotion = df['emotion_vector'].iloc[-1]
            if isinstance(last_emotion, dict):
                plot_emotion_radar(last_emotion, radar_path)

        # 心理状态摘要
        summary_text = ""
        if not df['emotion_vector'].empty:
            last_emotion = df['emotion_vector'].iloc[-1]
            dominant = last_emotion.get('dominant_emotion', 'unknown')
            confidence = last_emotion.get('confidence', 0.0)
            composite = [k for k in last_emotion.keys() if k not in [
                "happy", "sadness", "anger", "fear", "surprise", "disgust"
            ] and last_emotion[k] > 0.3]

            summary_text += f"【心理状态摘要】\n"
            summary_text += f"主导情绪: {dominant} (置信度 {confidence:.2f})\n"
            if composite:
                summary_text += f"复合情绪: {'、'.join(composite)}\n"
            if micro_times:
                summary_text += f"微表情活动: {len(micro_times)} 次\n"
            tension = df['tension_score'].iloc[-1] if not df.empty else 0
            if tension > 0.6:
                summary_text += "高紧张状态\n"
            elif tension > 0.3:
                summary_text += "中等紧张表现\n"
            else:
                summary_text += "当前情绪状态平稳\n"

        fig.text(0.95, 0.05, summary_text, fontsize=10, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

        print(f"✅ 可视化已保存至:\n - {save_img_path}\n - {radar_path}")
        return True

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else ""
    success = plot_features_from_csv(csv_path)
    if success:
        print("✅ 图表已成功显示并保存")
    else:
        print("❌ 图表生成失败")
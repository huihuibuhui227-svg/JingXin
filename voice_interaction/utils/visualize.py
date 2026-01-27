"""
语音交互日志可视化工具（整合版 + 递归查找）

功能：
- 自动在 data/logs 及其所有子目录中查找最新的 interview/research 日志
- 生成多维度分析图 + 特征热力图
- 自动保存到 data/output/voice_interaction/
- 支持中英文（自动降级）
- 空日志时生成示例数据

作者：镜心项目 - 语音分析模块
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# ==============================
# 1. 路径配置：优先使用 config，否则回退
# ==============================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试从 config 获取 LOGS_DIR
LOG_DIR = PROJECT_ROOT / "data" / "logs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "voice_interaction"

try:
    from voice_interaction.config import LOGS_DIR as CONFIG_LOGS_DIR
    LOG_DIR = Path(CONFIG_LOGS_DIR)
except (ImportError, AttributeError):
    print(f"⚠️ 使用默认日志目录: {LOG_DIR}")


# ==============================
# 2. 核心函数
# ==============================

def find_latest_log_file(log_type: str = 'auto') -> tuple[Path | None, str]:
    """
    递归查找 logs 目录下最新的 interview 或 research 日志（包括子目录）
    """
    log_dir = LOG_DIR
    if not log_dir.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return None, ''

    # 递归查找所有 CSV 文件
    all_csv_files = list(log_dir.rglob("*.csv"))

    interview_files = [f for f in all_csv_files if "interview_emotion_log_" in f.name]
    research_files = [f for f in all_csv_files if "research_emotion_log_" in f.name]

    if log_type == 'auto':
        if interview_files:
            latest = max(interview_files, key=os.path.getmtime)
            return latest, 'interview'
        elif research_files:
            latest = max(research_files, key=os.path.getmtime)
            return latest, 'research'
        else:
            return None, ''
    else:
        target_files = interview_files if log_type == 'interview' else research_files
        if not target_files:
            return None, log_type
        return max(target_files, key=os.path.getmtime), log_type


def ensure_output_dir():
    """确保输出目录存在"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_sample_data(n_questions: int = 10):
    """生成示例数据（用于空日志测试）"""
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    emotions = ['happy', 'neutral', 'sad', 'angry', 'fear']

    return pd.DataFrame({
        'unix_timestamp': [int(datetime.now().timestamp()) + i * 60 for i in range(n_questions)],
        'timestamp': [datetime.now() + pd.Timedelta(minutes=i) for i in range(n_questions)],
        'pitch_mean': np.random.uniform(150, 300, n_questions),
        'pitch_variation': np.random.uniform(10, 50, n_questions),
        'pitch_trend': np.random.choice(['increasing', 'decreasing', 'stable'], n_questions),
        'pitch_direction': np.random.choice(['up', 'down', 'stable'], n_questions),
        'energy_mean': np.random.uniform(0.1, 0.9, n_questions),
        'energy_variation': np.random.uniform(0.05, 0.3, n_questions),
        'speech_ratio': np.random.uniform(0.5, 0.95, n_questions),
        'duration_sec': np.random.uniform(10, 60, n_questions),
        'pause_duration_mean': np.random.uniform(0.5, 3.0, n_questions),
        'pause_duration_max': np.random.uniform(2.0, 8.0, n_questions),
        'pause_frequency': np.random.uniform(2, 10, n_questions),
        'emotion': np.random.choice(emotions, n_questions),
        'feedback': np.random.choice(['good', 'average', 'needs improvement'], n_questions),
        'question_index': list(range(1, n_questions + 1)),
        'is_valid': [True] * n_questions
    })


def setup_chinese_font():
    """设置中文字体，返回是否使用中文"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False, False

    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, '测试')
            plt.close(fig)
            return True, True
        except:
            continue

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return True, False


def plot_multidimensional_analysis(df, log_type: str, use_chinese: bool, output_path: Path):
    """绘制多维度分析图（6子图）"""
    import matplotlib.pyplot as plt

    def _(en, cn): return cn if use_chinese else en

    title_prefix = _("Interview Assessment Analysis", "面试评估分析") if log_type == 'interview' else \
                   _("Research Assessment Analysis", "科研评估分析")

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(title_prefix, fontsize=16)

    axes[0, 0].plot(df['question_index'], df['pitch_variation'], marker='o', linestyle='-', color='blue')
    axes[0, 0].set_title(_('Pitch Variation Over Questions', '音调变化随问题索引变化'))
    axes[0, 0].set_ylabel(_('Pitch Variation', '音调变化'))
    axes[0, 0].set_xlabel(_('Question Index', '问题索引'))
    axes[0, 0].grid(True)

    emotion_counts = df['emotion'].value_counts()
    axes[0, 1].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title(_('Emotion Distribution', '情绪状态分布'))

    x = range(len(df))
    width = 0.35
    axes[1, 0].bar([i - width/2 for i in x], df['speech_ratio'], width, label=_('Speech Ratio', '语音比例'), color='blue')
    axes[1, 0].bar([i + width/2 for i in x], df['energy_variation'], width, label=_('Energy Variation', '能量变化'), color='green')
    axes[1, 0].set_title(_('Voice Features Overview', '语音特征概览'))
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_xlabel(_('Question Index', '问题索引'))
    axes[1, 0].set_xticks(x)
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y')

    axes[1, 1].hist(df['duration_sec'], bins=20, color='purple', edgecolor='black')
    axes[1, 1].set_title(_('Response Duration Distribution', '回答持续时间分布'))
    axes[1, 1].set_xlabel('Duration (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, axis='y')

    axes[2, 0].plot(df['question_index'], df['pause_duration_mean'], marker='o', linestyle='-', color='red')
    axes[2, 0].set_title(_('Mean Pause Duration Over Questions', '平均停顿时间随问题索引变化'))
    axes[2, 0].set_ylabel(_('Mean Pause Duration (seconds)', '平均停顿时间（秒）'))
    axes[2, 0].set_xlabel(_('Question Index', '问题索引'))
    axes[2, 0].grid(True)

    axes[2, 1].plot(df['question_index'], df['pause_frequency'], marker='o', linestyle='-', color='orange')
    axes[2, 1].set_title(_('Pause Frequency Over Questions', '停顿频率随问题索引变化'))
    axes[2, 1].set_ylabel(_('Pause Frequency (pauses/min)', '停顿频率（次/分钟）'))
    axes[2, 1].set_xlabel(_('Question Index', '问题索引'))
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap(df, use_chinese: bool, output_path: Path):
    """绘制特征热力图"""
    import matplotlib.pyplot as plt

    features = ['pitch_mean', 'pitch_variation', 'energy_mean', 'energy_variation',
                'speech_ratio', 'duration_sec', 'pause_duration_mean', 'pause_frequency']
    available_features = [f for f in features if f in df.columns]

    if not available_features:
        raise ValueError("No numeric features found for heatmap")

    heatmap_data = df[available_features].values.T

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(df)))
    ax.set_yticks(range(len(available_features)))
    ax.set_yticklabels(available_features)
    ax.set_xlabel(('Question Index', '问题索引') if use_chinese else 'Question Index')
    ax.set_ylabel(('Features', '特征') if use_chinese else 'Features')
    ax.set_title(('Feature Heatmap', '特征热力图') if use_chinese else 'Feature Heatmap')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_voice_log(csv_path: Path, log_type: str):
    """主可视化函数"""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}\n请运行: pip install pandas matplotlib numpy")
        return False

    # 读取数据
    df = pd.read_csv(csv_path)
    if df.empty or len(df) <= 1:
        print("⚠️ 日志为空，生成示例数据...")
        df = generate_sample_data()

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 设置字体
    font_ok, use_chinese = setup_chinese_font()
    if not font_ok:
        return False

    # 确保输出目录
    ensure_output_dir()

    # 生成时间戳
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{log_type}_analysis_{timestamp_str}"

    # 1. 多维度分析图
    multi_path = OUTPUT_DIR / f"{base_name}_multidim.png"
    plot_multidimensional_analysis(df, log_type, use_chinese, multi_path)
    print(f"✅ 多维度分析图已保存: {multi_path}")

    # 2. 热力图
    heatmap_path = OUTPUT_DIR / f"{base_name}_heatmap.png"
    try:
        plot_heatmap(df, use_chinese, heatmap_path)
        print(f"✅ 特征热力图已保存: {heatmap_path}")
    except Exception as e:
        print(f"⚠️ 热力图生成失败: {e}")

    return True


def main():
    print("=" * 60)
    print("语音交互日志可视化（自动模式）")
    print("=" * 60)
    print(f"🔍 日志根目录: {LOG_DIR}")

    log_path, detected_type = find_latest_log_file('auto')

    if not log_path:
        print(f"❌ 未找到 interview_emotion_log_*.csv 或 research_emotion_log_*.csv")
        print("请先运行语音评估模块生成日志")
        # 列出目录内容帮助调试
        if LOG_DIR.exists():
            print("📁 日志目录中的所有 CSV 文件:")
            for f in LOG_DIR.rglob("*.csv"):
                print(f"  - {f}")
        return

    print(f"📄 检测到日志: {log_path.name} (类型: {detected_type})")
    success = visualize_voice_log(log_path, detected_type)

    print("=" * 60)
    if success:
        print("✅ 所有图表已成功生成并保存")
    else:
        print("❌ 可视化失败")
    print("=" * 60)


if __name__ == "__main__":
    main()
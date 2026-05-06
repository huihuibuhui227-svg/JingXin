import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

# === 配置路径处理 (修复 mkdir 报错) ===
FACE_EXPRESSION_OUTPUT_DIR = None

try:
    from ..config import FACE_EXPRESSION_OUTPUT_DIR as config_path

    FACE_EXPRESSION_OUTPUT_DIR = Path(config_path) if not isinstance(config_path, Path) else config_path
except ImportError:
    import sys

    config_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(config_dir))
    try:
        from config import FACE_EXPRESSION_OUTPUT_DIR as config_path

        FACE_EXPRESSION_OUTPUT_DIR = Path(config_path) if not isinstance(config_path, Path) else config_path
    except ImportError as e:
        print(f"⚠️ 无法导入配置文件：{e}")
        project_root = config_dir.parent
        FACE_EXPRESSION_OUTPUT_DIR = project_root / 'data' / 'output' / 'face_expression'

# 确保目录存在 (修复 'str' object has no attribute 'mkdir')
if FACE_EXPRESSION_OUTPUT_DIR:
    FACE_EXPRESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
else:
    # 兜底策略
    FACE_EXPRESSION_OUTPUT_DIR = Path.cwd() / 'data' / 'output' / 'face_expression'
    FACE_EXPRESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if parent.name == 'jingxin':
            return parent
    return Path(__file__).parent.parent.parent  # 兜底返回


def get_latest_log_file(log_dir: str = "data/logs", prefix: str = "face_au_log") -> str:
    """获取最新日志文件"""
    project_root = get_project_root()
    log_path = project_root / log_dir

    if not log_path.exists():
        # 尝试相对路径
        log_path = Path(log_dir)

    if not log_path.exists():
        print(f"❌ 日志目录 {log_path} 不存在")
        return ""

    csv_files = list(log_path.glob(f"{prefix}_*.csv"))
    if not csv_files:
        print(f"❌ 未找到以 {prefix}_ 开头的日志文件")
        return ""

    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    print(f"✅ 找到最新日志文件：{latest_file.name}")
    return str(latest_file)


def _safe_json_loads(s: str) -> Dict[str, Any]:
    """安全解析 JSON"""
    if pd.isna(s) or s == '{}' or s == '':
        return {}
    try:
        return json.loads(str(s).replace("'", '"'))
    except:
        return {}


def plot_features_from_csv(csv_path: str = "") -> bool:
    """
    增强型可视化：自动适配 JSON 列 或 扁平化列
    """
    if not csv_path:
        csv_path = get_latest_log_file()
        if not csv_path:
            return False

    if not os.path.exists(csv_path):
        print(f"❌ 文件 {csv_path} 不存在")
        return False

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("❌ CSV 为空")
            return False

        print(f"📊 原始数据行数：{len(df)}")

        # === 核心修复：智能检测列类型 ===
        has_json_cols = 'psychological_signals' in df.columns

        if has_json_cols:
            print("ℹ️ 检测到 JSON 格式列，正在解析...")
            # 旧版逻辑：解析 JSON
            df['psychological_signals'] = df['psychological_signals'].apply(_safe_json_loads)
            df['micro_expressions'] = df['micro_expressions'].apply(_safe_json_loads)
            df['emotion_vector'] = df['emotion_vector'].apply(_safe_json_loads)

            df['tension_score'] = df['psychological_signals'].apply(lambda x: x.get('tension_score', 0))
            df['dominant_emotion'] = df['emotion_vector'].apply(lambda x: x.get('dominant_emotion', 'unknown'))
            df['confidence'] = df['emotion_vector'].apply(lambda x: x.get('confidence', 0))

            # 微表情检测 (JSON 模式)
            micro_mask = df['micro_expressions'].apply(lambda x: x != {})
        else:
            print("ℹ️ 检测到扁平化列，直接使用...")
            # 新版逻辑：直接使用列
            # 确保关键列存在，不存在则创建默认值防止报错
            if 'tension_score' not in df.columns: df['tension_score'] = 0
            if 'dominant_emotion' not in df.columns: df['dominant_emotion'] = 'unknown'
            if 'confidence' not in df.columns: df['confidence'] = 0

            # 微表情检测 (扁平模式：检查 micro_exp_au_name 是否非空)
            if 'micro_exp_au_name' in df.columns:
                micro_mask = df['micro_exp_au_name'].notna() & (df['micro_exp_au_name'] != '')
            else:
                micro_mask = pd.Series([False] * len(df))

        # 清洗空行
        initial_len = len(df)
        df = df.dropna(how='all')
        if len(df) < initial_len:
            print(f"✨ 清洗后有效数据行数：{len(df)} (移除 {initial_len - len(df)} 空行)")

        micro_times = df[micro_mask].index.tolist()
        if micro_times:
            print(f"ℹ️ 检测到 {len(micro_times)} 个微表情，将在图中稀疏显示。")

        # 时间轴
        if 'timestamp' in df.columns:
            time_sec = (df['timestamp'] - df['timestamp'].iloc[0])
        else:
            time_sec = df.index / 30.0

        # 设置字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle("面部心理状态综合分析", fontsize=16)

        # 1. 紧张度
        ax = axes[0, 0]
        ax.plot(time_sec, df['tension_score'], 'r-', label='紧张度', linewidth=1.5)
        if micro_times:
            step = max(1, len(micro_times) // 20)
            for i, t_idx in enumerate(micro_times[::step]):
                if 0 <= t_idx < len(time_sec):
                    label = '微表情' if i == 0 else ""
                    ax.axvline(x=time_sec.iloc[t_idx], color='purple', linestyle='--', alpha=0.6, label=label)
        ax.set_ylabel('紧张度 (0-1)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # 2. 置信度
        ax = axes[0, 1]
        ax.plot(time_sec, df['confidence'], 'b-', label='情绪置信度', linewidth=1.5)
        ax.set_ylabel('置信度')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # 3. 基础 AU
        ax = axes[1, 0]
        au4_col = 'au4_frown' if 'au4_frown' in df.columns else None
        au12_col = 'au12_smile' if 'au12_smile' in df.columns else None
        if au4_col: ax.plot(time_sec, df[au4_col], label='AU4 (皱眉)', linewidth=1.2)
        if au12_col: ax.plot(time_sec, df[au12_col], label='AU12 (微笑)', linewidth=1.2)
        if not au4_col and not au12_col: ax.text(0.5, 0.5, '无 AU 数据', ha='center', transform=ax.transAxes)
        ax.set_ylabel('AU 强度')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 4. 其他 AU
        ax = axes[1, 1]
        au6_col = 'au6_cheek_raise' if 'au6_cheek_raise' in df.columns else None
        au23_col = 'au23_lip_compression' if 'au23_lip_compression' in df.columns else None
        if au6_col: ax.plot(time_sec, df[au6_col], label='AU6 (脸颊抬起)', linewidth=1.2)
        if au23_col: ax.plot(time_sec, df[au23_col], label='AU23 (嘴唇压缩)', linewidth=1.2)
        if not au6_col and not au23_col: ax.text(0.5, 0.5, '无 AU 数据', ha='center', transform=ax.transAxes)
        ax.set_ylabel('AU 强度')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 5. 眨眼
        ax = axes[2, 0]
        blink_col = 'blink_rate_per_min' if 'blink_rate_per_min' in df.columns else None
        if blink_col and df[blink_col].sum() > 0:
            ax.plot(time_sec, df[blink_col], 'k-', label='眨眼频率')
        else:
            ax.text(0.5, 0.5, '无眨眼数据', ha='center', transform=ax.transAxes)
        ax.set_ylabel('眨眼/分钟')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 6. 视线 (修复 UserWarning)
        ax = axes[2, 1]
        gaze_x_col = 'gaze_direction_x' if 'gaze_direction_x' in df.columns else None
        gaze_y_col = 'gaze_direction_y' if 'gaze_direction_y' in df.columns else None

        if gaze_x_col and gaze_y_col:
            ax.scatter(time_sec, df[gaze_x_col], label='视线 X', c='red', s=10, alpha=0.6)  # 使用 c 而不是 color
            ax.scatter(time_sec, df[gaze_y_col], label='视线 Y', c='blue', s=10, alpha=0.6)
            ax.set_ylabel('视线方向')
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, '无视线数据', ha='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # 7. 主导情绪 (修复 KeyError 和 空数据)
        ax = axes[3, 0]
        if 'dominant_emotion' in df.columns:
            unique_emo = df['dominant_emotion'].dropna().unique()
            if len(unique_emo) > 0:
                emo_map = {e: i for i, e in enumerate(unique_emo)}
                # 映射颜色
                colors = [plt.cm.tab10(emo_map[e] % 10) for e in df['dominant_emotion']]
                # 处理 NaN 对应的颜色
                colors = [c if pd.notna(e) else 'gray' for e, c in zip(df['dominant_emotion'], colors)]

                ax.scatter(time_sec, [0.5] * len(df), c=colors, s=20, alpha=0.8)

                # 添加图例
                for e, idx in emo_map.items():
                    ax.scatter([], [], c=plt.cm.tab10(idx % 10), label=str(e), s=30)
                ax.legend(loc='upper right', fontsize=8)
            else:
                ax.text(0.5, 0.5, '无有效情绪数据', ha='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, '无情绪列', ha='center', transform=ax.transAxes)
        ax.set_yticks([])
        ax.set_ylabel('主导情绪')
        ax.grid(True, alpha=0.3)

        # 8. 末帧雷达图 (简化版，避免复杂报错)
        ax = axes[3, 1]
        basic_emotions = ["happy", "sadness", "anger", "fear", "surprise", "disgust"]
        # 尝试获取最后一行数据
        last_row = df.iloc[-1]
        values = []
        has_data = False

        # 优先尝试从扁平列获取
        if all(e in df.columns for e in basic_emotions):
            values = [float(last_row[e]) for e in basic_emotions]
            has_data = True
        # 其次尝试从 JSON 列获取 (如果存在)
        elif 'emotion_vector' in df.columns and isinstance(last_row['emotion_vector'], dict):
            vec = last_row['emotion_vector']
            values = [vec.get(e, 0) for e in basic_emotions]
            has_data = True

        if has_data and any(v > 0 for v in values):
            angles = np.linspace(0, 2 * np.pi, len(basic_emotions), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            ax.plot(angles, values, 'o-', color='blue', linewidth=2)
            ax.fill(angles, values, alpha=0.25, color='lightblue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(basic_emotions, rotation=30, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_title('末帧情绪雷达图')
        else:
            ax.text(0.5, 0.5, '无详细情绪分数\n(happy, sadness...)', ha='center', va='center', transform=ax.transAxes,
                    fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存
        log_filename = Path(csv_path).stem
        save_img_path = FACE_EXPRESSION_OUTPUT_DIR / f'{log_filename}_enhanced_analysis.png'

        try:
            plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
            print(f"✅ 可视化已保存至:\n - {save_img_path}")
            plt.show()
            return True
        except Exception as e:
            print(f"❌ 保存失败：{e}")
            return False

    except Exception as e:
        print(f"❌ 可视化执行失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else ""
    success = plot_features_from_csv(csv_path)
    if success:
        print("✅ 流程结束")
    else:
        print("❌ 流程异常终止")
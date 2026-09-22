# report_frontend/data_loader.py

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import re
import urllib.request
import urllib.error


class LogDataLoader:
    """
    多模态日志数据加载器 (方案 A：跨时间模态融合版)

    策略：
    1. 递归扫描 data/logs 及其所有子目录。
    2. 将文件按模态 (face/gesture/interview/research) 分组。
    3. 在每组中选择时间戳最新的文件。
    4. 加载并返回这些“各自最新”的文件，组成一个完整的分析数据集。
    """

    def __init__(self, log_dir: str = None):
        """
        初始化加载器
        :param log_dir: 可选，自定义日志根目录。默认自动推导至项目根目录/data/logs
        """
        if log_dir is None:
            # 动态路径推导：当前文件 -> report_frontend -> jingxin (根目录) -> data/logs
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            self.log_dir = project_root / "data" / "logs"
        else:
            self.log_dir = Path(log_dir)

        if not self.log_dir.exists():
            raise FileNotFoundError(
                f"❌ 错误：日志目录不存在！\n"
                f"   尝试路径：{self.log_dir}\n"
                f"   请检查目录结构或手动指定 log_dir。"
            )

        print(f"✅ 日志根目录已定位：{self.log_dir}")

        # 正则表达式匹配文件名
        # 格式：{type}_{desc}_log_{YYYYMMDD}_{HHMMSS}.csv
        self.file_pattern = re.compile(r"^(face|gesture|interview|research)_(.+?)_log_(\d{8})_(\d{6})\.csv$")

    def _scan_and_group_files(self) -> Dict[str, List[Dict]]:
        """
        内部方法：递归扫描所有 CSV，并按模态分组
        :return: {'face': [file_info, ...], 'gesture': [...], ...}
        """
        groups = {
            'face': [],
            'gesture': [],
            'interview': [],
            'research': []
        }

        # 使用 rglob 递归查找所有子目录下的 csv
        all_csvs = list(self.log_dir.rglob("*.csv"))

        for file_path in all_csvs:
            if not file_path.is_file():
                continue

            match = self.file_pattern.match(file_path.name)
            if match:
                modality = match.group(1)
                date_str = match.group(3)
                time_str = match.group(4)
                # 生成用于比较的整数时间戳 YYYYMMDDHHMMSS
                timestamp_val = int(f"{date_str}{time_str}")

                file_info = {
                    "path": file_path,
                    "session_id": f"{date_str}_{time_str}",
                    "timestamp_val": timestamp_val
                }

                if modality in groups:
                    groups[modality].append(file_info)

        return groups

    def get_fused_latest_data(self) -> Dict[str, pd.DataFrame]:
        """
        【主入口】获取各模态最新文件的融合数据
        :return: {'face': df, 'gesture': df, 'voice_interview': df, 'voice_research': df}
        """
        print("\n🔍 正在扫描并筛选各模态的最新日志...")
        print("-" * 70)

        groups = self._scan_and_group_files()
        selected_files = []

        # 遍历每个模态组，选出时间最新的一个
        for modality, files in groups.items():
            if not files:
                print(f"   ⚠️  [{modality.upper()}] 未找到相关日志文件。")
                continue

            # 按时间戳排序，取最后一个（最新）
            latest_file = max(files, key=lambda x: x['timestamp_val'])
            selected_files.append({
                "modality": modality,
                "info": latest_file
            })

            rel_path = latest_file['path'].relative_to(self.log_dir)
            print(f"   ✅ [{modality.upper()}] 选中最新文件：{rel_path}")
            print(f"       会话时间：{latest_file['session_id']}")

        if not selected_files:
            print("-" * 70)
            print("❌ 错误：未发现任何符合命名规范的日志文件。")
            return {}

        print("-" * 70)
        print(f"🚀 开始加载 {len(selected_files)} 个模态数据...")

        data_frames = {}

        for item in selected_files:
            modality = item['modality']
            file_info = item['info']
            file_path = file_info['path']

            # 统一模态键名，方便后续处理
            if modality == 'interview':
                key = 'voice_interview'
            elif modality == 'research':
                key = 'voice_research'
            else:
                key = modality

            try:
                # 自动识别编码读取 CSV
                df = self._read_csv_safe(file_path)

                if df is None or df.empty:
                    print(f"   ⚠️  [{key.upper()}] 文件为空或读取失败。")
                    continue

                # --- 数据标准化处理 ---
                df = self._normalize_dataframe(df)

                data_frames[key] = df
                print(f"   📥 [{key.upper()}] 加载成功：{len(df)} 行，{len(df.columns)} 列")

                # 打印关键列预览
                key_cols = [c for c in ['timestamp', 'au_1', 'au_12', 'emotion', 'score', 'jitter', 'fluency'] if
                            c in df.columns]
                if key_cols:
                    print(f"      🔑 关键列检测：{key_cols}")

            except Exception as e:
                print(f"   ❌ [{key.upper()}] 加载过程中发生异常：{e}")
                import traceback
                traceback.print_exc()

        print("-" * 70)

        # 【修复点 1】这里补全了 if data_frames:
        if data_frames:
            print(f"🎉 数据融合完成！可用模态：{list(data_frames.keys())}")
        else:
            print("💥 最终结果：没有成功加载任何有效数据。")

        return data_frames

    def get_live_data(self, session_id: str,
                       face_port: int = 8000,
                       gesture_port: int = 8002,
                       voice_port: int = 8001,
                       timeout: int = 5) -> Dict[str, pd.DataFrame]:
        """【实时数据源】通过 HTTP 从三个 API 获取指定会话的内存数据。

        :param session_id: 目标会话 ID
        :param face_port: 面部分析 API 端口 (默认 8000)
        :param gesture_port: 手势分析 API 端口 (默认 8002)
        :param voice_port: 语音交互 API 端口 (默认 8001)
        :param timeout: 单个请求超时秒数 (默认 5)
        :return: 与 get_fused_latest_data() 同格式的 DataFrame 字典
        """
        data_frames: Dict[str, pd.DataFrame] = {}

        # --- 面部数据 ---
        try:
            url = f"http://127.0.0.1:{face_port}/session/{session_id}/summary"
            face_json = self._http_get_json(url, timeout)
            if face_json and face_json.get("status") == "success":
                summary = face_json["data"]
                # 将聚合统计数据展平为单行 DataFrame
                row: Dict[str, float] = {}
                row["frame_count"] = summary.get("frame_count", 0)
                row["duration_sec"] = summary.get("duration_sec", 0)
                row["tension_score"] = summary.get("tension", {}).get("avg_score", 0)
                row["focus_score"] = summary.get("focus", {}).get("avg_score", 0.5)
                row["gaze_deviation"] = summary.get("gaze", {}).get("avg_deviation", 0)
                row["gaze_stability"] = summary.get("gaze", {}).get("stability", 0.8)
                row["symmetry_score"] = summary.get("au_features", {}).get("symmetry_score", {}).get("mean", 1.0)
                # 展开各 AU 均值
                for au_name, au_stat in summary.get("au_features", {}).items():
                    row[au_name] = au_stat.get("mean", 0)
                # 展开情绪分布
                for emo, count in summary.get("emotion", {}).get("distribution", {}).items():
                    row[f"emotion_{emo}"] = count
                # 眨眼
                row["blink_rate_per_min"] = summary.get("blink", {}).get("recent_blinks_per_min", 0)
                df = pd.DataFrame([row])
                df["timestamp"] = pd.Timestamp.now()
                data_frames["face"] = df
                print(f"   📥 [FACE] 实时数据加载成功：{summary.get('frame_count', 0)} 帧")
        except Exception as e:
            print(f"   ⚠️  [FACE] 实时数据获取失败: {e}")

        # --- 手势数据 ---
        try:
            url = f"http://127.0.0.1:{gesture_port}/session/{session_id}/summary"
            gesture_json = self._http_get_json(url, timeout)
            if gesture_json and gesture_json.get("status") == "success":
                gd = gesture_json["data"]
                row: Dict[str, float] = {}
                row["hand_score"] = gd.get("hand", {}).get("average_score", 50)
                row["left_hand_score"] = gd.get("hand", {}).get("left", {}).get("resilience_score", 50)
                row["right_hand_score"] = gd.get("hand", {}).get("right", {}).get("resilience_score", 50)
                row["hand_jitter"] = gd.get("hand", {}).get("left", {}).get("jitter", 0)
                row["shoulder_score"] = gd.get("shoulder", {}).get("shoulder_score", 50)
                row["shrug_level"] = gd.get("shoulder", {}).get("shrug_level", 0)
                row["left_arm_score"] = gd.get("arm", {}).get("left", {}).get("arm_score", 50)
                row["right_arm_score"] = gd.get("arm", {}).get("right", {}).get("arm_score", 50)
                row["emotion_score"] = gd.get("emotion", {}).get("overall_score", 50)
                row["emotion_state"] = gd.get("emotion", {}).get("emotion_state", "neutral")
                df = pd.DataFrame([row])
                df["timestamp"] = pd.Timestamp.now()
                data_frames["gesture"] = df
                print(f"   📥 [GESTURE] 实时数据加载成功")
        except Exception as e:
            print(f"   ⚠️  [GESTURE] 实时数据获取失败: {e}")

        # --- 语音数据（面试） ---
        try:
            url = f"http://127.0.0.1:{voice_port}/session/{session_id}/summary?type=interview"
            voice_json = self._http_get_json(url, timeout)
            if voice_json and voice_json.get("status") == "success":
                vd = voice_json["data"]
                qa_pairs = vd.get("qa_pairs", [])
                if qa_pairs:
                    rows = []
                    for qa in qa_pairs:
                        answer = qa.get("answer", "")
                        rows.append({
                            "question": qa.get("question", ""),
                            "answer": answer,
                            "answer_length": len(answer),
                            "has_valid_answer": qa.get("has_valid_answer", False),
                        })
                    df = pd.DataFrame(rows)
                    df["timestamp"] = pd.Timestamp.now()
                    df["evaluation"] = vd.get("evaluation", "")
                    data_frames["voice_interview"] = df
                    print(f"   📥 [VOICE_INTERVIEW] 实时数据加载成功：{len(qa_pairs)} 个问答")
                else:
                    print(f"   ⚠️  [VOICE_INTERVIEW] 暂无问答数据")
        except Exception as e:
            print(f"   ⚠️  [VOICE_INTERVIEW] 实时数据获取失败: {e}")

        return data_frames

    @staticmethod
    def _http_get_json(url: str, timeout: int = 5) -> Optional[Dict]:
        """HTTP GET 请求，返回解析后的 JSON。"""
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            print(f"      HTTP {e.code}: {body[:200]}")
            return None
        except Exception as e:
            print(f"      HTTP 请求异常: {e}")
            return None

    def _read_csv_safe(self, file_path: Path) -> Optional[pd.DataFrame]:
        """安全读取 CSV，尝试多种编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except (UnicodeDecodeError, ValueError):
                continue
        return None

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化 DataFrame：
        1. 确保有 'timestamp' 列 (datetime 类型)
        2. 按时间排序
        3. 重置索引
        """
        # 常见时间列名映射
        time_candidates = ['timestamp', 'time', 'datetime', 'record_time', 'timestamp_iso', 'create_time']
        found_col = None

        for col in time_candidates:
            if col in df.columns:
                found_col = col
                break

        if found_col:
            # 转换为 datetime，如果失败则 coerce 为 NaT
            df['timestamp'] = pd.to_datetime(df[found_col], errors='coerce')
            # 如果转换后全是 NaT，尝试将其作为数值时间戳处理
            if df['timestamp'].isna().all():
                df['timestamp'] = pd.to_numeric(df[found_col], errors='coerce')
        else:
            # 如果没有时间列，使用索引
            df['timestamp'] = pd.to_numeric(df.index, errors='coerce')

        # 排序并重置索引
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

        return df

    def get_available_modalities_summary(self) -> str:
        """快速概览有哪些模态数据可用"""
        groups = self._scan_and_group_files()
        summary = []
        for mod, files in groups.items():
            if files:
                latest = max(files, key=lambda x: x['timestamp_val'])
                summary.append(f"{mod}: {latest['session_id']} ({len(files)} 个文件)")
            else:
                summary.append(f"{mod}: 无数据")
        return "\n".join(summary)


# --- 本地测试入口 ---
if __name__ == "__main__":
    print("=== 启动 Data Loader (方案 A：跨时间融合模式) ===")
    try:
        loader = LogDataLoader()

        # 打印概览
        print("\n📊 当前数据概览:")
        print(loader.get_available_modalities_summary())

        # 执行融合加载
        data = loader.get_fused_latest_data()

        # 【修复点 2】这里补全了 if data:
        if data:
            print("\n✅ 测试通过！已成功加载融合数据。")
            for key, df in data.items():
                t_start = df['timestamp'].min()
                t_end = df['timestamp'].max()
                print(f"   - {key}: {len(df)} 条记录 (时间跨度：{t_start} ~ {t_end})")
        else:
            print("\n⚠️ 未加载到任何数据，请检查 logs 目录。")

    except Exception as e:
        print(f"\n💥 程序运行出错：{e}")
        import traceback

        traceback.print_exc()
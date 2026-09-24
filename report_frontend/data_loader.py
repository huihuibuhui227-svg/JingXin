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

        # 本轮真正加载到的模态 → 那个文件自报的 session_id。选择策略本身仍是"每模态取最新"
        # (按文件名时间戳,不看这一列;按 id 选是 M2 的事),这里只把它**交出来** ——
        # 报告头据此披露"这份报告由哪些日志装配而成"。在这之前,加载器算出的 session_id
        # 唯一的消费者是两处 `print()`,于是跨场拼接在报告里是隐形的。
        self.selected_sessions: Dict[str, str] = {}

        # 正则表达式匹配文件名。**两种形态都接受**,报告侧只认文件名里的时间戳
        # (不读 mtime、不读内容、也不读 session_id 列):
        #   旧形态  {type}_{desc}_log_{YYYYMMDD}_{HHMMSS}.csv              —— 历史日志
        #   M1 形态 {type}_{desc}_log_{YYYYMMDD}_{HHMMSS}_{4 位十六进制}.csv
        #           session_id = new_session_id()(voice_interaction/asr/session.py):
        #           时间段给人看,随机段防撞。T3 起三个模块的文件名都带它
        #           (face/gesture 的 API、voice 的 logger),即**没有产出方**会再写出
        #           纯时间戳形态 —— 只认旧形态会让报告路径一份日志都加载不到。
        # ⚠️ 紧跟在 `_log_` 后面的**时间戳一段不是可选的**,NONE 桶因此进不来:
        #   无 id 会话落盘为 `..._log_NONE_{YYYYMMDD}_{HHMMSS}.csv`(三个 API 都把缺省 id
        #   解析成字面量 "NONE" 后传给 logger),它的尾部**同样长得像时间戳**。若把时间戳
        #   放宽成"可选",这些跨天增长的文件会重新可见,一次聚合就会把不同天、不同客户端的
        #   行混在一起(此前"它们从不进入任何聚合"这一保证正依赖于它们不可见)。
        self.file_pattern = re.compile(
            r"^(face|gesture|interview|research)_(.+?)_log_(\d{8})_(\d{6})"
            r"(?:_([0-9a-f]{4}))?\.csv$"
        )

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
                sid_suffix = match.group(5)          # M1 形态的随机段;旧形态为 None
                # 生成用于比较的整数时间戳 YYYYMMDDHHMMSS(随机段不参与排序)
                timestamp_val = int(f"{date_str}{time_str}")

                file_info = {
                    "path": file_path,
                    # 自报的会话 = 文件名里那一段(旧形态只有时间戳,M1 形态带随机段)
                    "session_id": f"{date_str}_{time_str}" + (f"_{sid_suffix}" if sid_suffix else ""),
                    "timestamp_val": timestamp_val
                }

                if modality in groups:
                    groups[modality].append(file_info)

        return groups

    NONE_SESSION = "NONE"

    def _none_bucket_rows(self) -> int:
        """数 NONE 桶里有多少行。

        NONE 文件**刻意不被 `file_pattern` 匹配**(见那段注释):它们不进任何聚合。
        但"存在这样一批行"这件事本身要告诉读者 —— 否则有帧没归入本场而无人知晓。
        """
        total = 0
        for p in self.log_dir.rglob("*_log_NONE_*.csv"):
            try:
                df = self._read_csv_safe(p)
                total += 0 if df is None else len(df)
            except Exception:
                pass
        return total

    def resolve_target_session(self, session_id: Optional[str] = None) -> Optional[str]:
        """定这一场报告以哪个 `session_id` 为准(M2 spec §5.1)。

        * 显式给了 -> 就用它(`NONE` 除外 —— 那不是一场会话,是"没给 id"的占位)
        * 没给     -> 取**文件名时间戳最大**的那一场的 id
        * 一份都没有 -> `None`。调用方据此说"本场没有任何日志",**不许回退去拼别的场次**

        为什么不能沿用"每模态各取最新":那正是把三场会话拼在一起的机制
        (2026-09-24 真实前端使用实测 → 报告恒为「0 / 20」,spec §3.1)。
        """
        if session_id:
            return None if session_id == self.NONE_SESSION else session_id

        candidates = [f for files in self._scan_and_group_files().values() for f in files
                      if f["session_id"] != self.NONE_SESSION]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x["timestamp_val"])["session_id"]

    def get_fused_latest_data(self, session_id: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """【主入口】按**目标会话**取各模态日志。

        M2 起:`session_id` 定了之后,**每个模态只找该 id 的文件** —— 不再"每模态各取最新"
        (那会把不同场次拼在一起)。每个模态的结果是三态之一,写进 `self.selected_sessions`:

            loaded / unreadable(文件在但读不出) / missing(本场没有这个模态)

        ⚠️ `selected_sessions` 的值**从 `str` 变成了 `dict`** —— 这是破坏性变更,
        所有消费点(`report_generator.sources_disclosure` 等)必须一起改。
        """
        print("\n🔍 按 session_id 取日志...")
        print("-" * 70)

        target = self.resolve_target_session(session_id)
        self.selected_sessions = {}
        if target is None:
            print("❌ 没有可用于本报告的会话(只有 NONE 桶,或没有任何符合命名规范的日志)。")
            return {}
        print(f"   目标会话:{target}")

        none_rows = self._none_bucket_rows()
        if none_rows:
            self.selected_sessions["none_bucket"] = {
                "session_id": self.NONE_SESSION, "status": "present", "rows": none_rows}
            print(f"   ℹ️  另有 NONE 桶 {none_rows} 行(未归入任何会话,不进聚合)")

        print("-" * 70)

        data_frames = {}
        groups = self._scan_and_group_files()

        for modality, files in groups.items():
            # 统一模态键名，方便后续处理
            key = {'interview': 'voice_interview',
                   'research': 'voice_research'}.get(modality, modality)

            mine = [f for f in files if f["session_id"] == target]
            if not mine:
                self.selected_sessions[key] = {"session_id": target, "status": "missing", "rows": 0}
                print(f"   ⚠️  [{key.upper()}] 本场没有这个模态")
                continue

            # 同一会话同一模态理论上只有一份;万一有多份,取时间戳最大的那份
            chosen = max(mine, key=lambda x: x["timestamp_val"])
            try:
                df = self._read_csv_safe(chosen["path"])
                if df is None or df.empty:
                    self.selected_sessions[key] = {"session_id": target,
                                                   "status": "unreadable", "rows": 0}
                    print(f"   ⚠️  [{key.upper()}] 文件在但读不出(空或损坏):{chosen['path'].name}")
                    continue

                df = self._normalize_dataframe(df)
                data_frames[key] = df
                self.selected_sessions[key] = {"session_id": target,
                                               "status": "loaded", "rows": len(df)}
                print(f"   📥 [{key.upper()}] 加载成功：{len(df)} 行，{len(df.columns)} 列")

            except Exception as e:
                self.selected_sessions[key] = {"session_id": target,
                                               "status": "unreadable", "rows": 0}
                print(f"   ❌ [{key.upper()}] 读取出错：{e}")

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
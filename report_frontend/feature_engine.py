# report_frontend/feature_engine.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import warnings
import re

warnings.filterwarnings('ignore')


class PsychologicalFeatureEngine:
    """
    心理特征提取引擎 (最终增强版·科研数据专用)

    【功能定位】
    本模块仅负责从多模态日志中提取高维量化特征向量。
    不进行任何心理状态解读或能力映射，确保数据的客观性与完整性。
    输出结果将直接作为下游模型 (Research Mapper) 的输入。

    【处理策略】
    1. 命名规范化：自动清洗键名，消除冗余符号，统一格式。
    2. 类型自适应：数值列计算统计矩，文本列 (AU/语音) 计算频率与密度。
    3. 样本感知：大样本计算时间趋势，小样本聚焦全量统计。
    4. 眼动专项分析：针对 iris/gaze 列计算稳定性、注视中心度等指标。
    5. 智能别名映射：为语音特征生成标准关键词别名，确保 Mapper 精准匹配。
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.features = {}

    def extract_all_features(self) -> Dict[str, Any]:
        print("\n⚙️ 正在执行全量特征提取 (纯净模式)...")
        print("-" * 70)

        try:
            # 1. 面部特征
            face_features = self._extract_face_features()
            if face_features:
                self.features['face'] = self._normalize_keys(face_features)
                print(f"   ✅ [FACE] 提取完成：{len(self.features['face'])} 个标准化特征")

            # 2. 手势特征
            gesture_features = self._extract_gesture_features()
            if gesture_features:
                self.features['gesture'] = self._normalize_keys(gesture_features)
                print(f"   ✅ [GESTURE] 提取完成：{len(self.features['gesture'])} 个标准化特征")

            # 3. 语音特征 (Interview & Research)
            for key in ['voice_interview', 'voice_research']:
                if key in self.data:
                    voice_features = self._extract_voice_features(key)
                    if voice_features:
                        self.features[key] = self._normalize_keys(voice_features)
                        print(f"   ✅ [{key.upper()}] 提取完成：{len(self.features[key])} 个标准化特征")

        except Exception as e:
            print(f"❌ 特征提取发生严重错误：{e}")
            import traceback
            traceback.print_exc()
            return {}

        print("-" * 70)
        total_count = sum(len(f) for f in self.features.values())
        print(f"🎯 提取完毕。总计生成 {total_count} 个标准化量化指标。")
        return self.features

    # ---------------------------------------------------------
    # 核心工具：键名规范化 (Naming Normalization)
    # ---------------------------------------------------------
    def _normalize_keys(self, features: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for k, v in features.items():
            clean_k = k.lower()
            clean_k = re.sub(r'_+', '_', clean_k)
            clean_k = clean_k.strip('_')
            if clean_k:
                normalized[clean_k] = v
        return normalized

    # ---------------------------------------------------------
    # 核心工具：AU 字符串解析
    # ---------------------------------------------------------
    def _parse_au_string_column(self, series: pd.Series, col_name: str) -> Dict[str, float]:
        au_pattern = re.compile(r'au(\d+)', re.IGNORECASE)
        au_counts = {}
        total_rows = len(series)
        if total_rows == 0: return {}

        valid_rows = 0
        complexity_sum = 0
        for text in series.dropna():
            if not isinstance(text, str): continue
            valid_rows += 1
            found_aus = au_pattern.findall(text)
            complexity_sum += len(found_aus)
            for au_num in found_aus:
                key = f"au{au_num}"
                au_counts[key] = au_counts.get(key, 0) + 1

        if valid_rows == 0: return {}
        features = {}
        for au_key, count in au_counts.items():
            features[f"{col_name}_{au_key}_freq"] = count / valid_rows
        features[f"{col_name}_avg_complexity"] = complexity_sum / valid_rows
        return features

    # ---------------------------------------------------------
    # 核心工具：数值统计提取
    # ---------------------------------------------------------
    def _extract_numeric_stats(self, df: pd.DataFrame, prefix: str, min_rows_for_trend: int = 20) -> Dict[str, float]:
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if any(x in col.lower() for x in ['id', 'index', 'unnamed', 'timestamp']):
                continue
            series = df[col].dropna()
            if len(series) == 0: continue

            if col.lower() in prefix.lower():
                base_name = prefix.rstrip('_')
            else:
                base_name = f"{prefix}{col}"

            stats[f"{base_name}_mean"] = float(series.mean())
            stats[f"{base_name}_std"] = float(series.std()) if len(series) > 1 else 0.0
            stats[f"{base_name}_min"] = float(series.min())
            stats[f"{base_name}_max"] = float(series.max())
            stats[f"{base_name}_sum"] = float(series.sum())

            if len(series) >= min_rows_for_trend:
                n = len(series)
                start_avg = series.iloc[:max(1, n // 10)].mean()
                end_avg = series.iloc[-max(1, n // 10):].mean()
                stats[f"{base_name}_trend"] = float(end_avg - start_avg)
            else:
                stats[f"{base_name}_sample_size"] = len(series)
        return stats

    # ---------------------------------------------------------
    # 1. 面部特征提取 (增强版：含眼动专项分析)
    # ---------------------------------------------------------
    def _extract_face_features(self) -> Optional[Dict[str, Any]]:
        if 'face' not in self.data or self.data['face'].empty:
            return None

        df = self.data['face']
        features = {}
        target_keywords = ['score', 'tension', 'anxiety', 'focus', 'symmetry', 'happy', 'sad', 'au_', 'AU']

        # --- A. 常规数值列与 AU 提取 ---
        for col in df.columns:
            col_lower = col.lower()
            if 'timestamp' in col_lower or 'file' in col_lower or 'path' in col_lower:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if any(k in col_lower for k in target_keywords):
                    feats = self._extract_numeric_stats(df[[col]], prefix=f"face_{col}_")
                    features.update(feats)

            elif 'au' in col_lower:
                au_stats = self._parse_au_string_column(df[col], col_name=f"face_micro_exp_{col}")
                features.update(au_stats)

        # --- B. 【核心】眼动数据专项提取 (Gaze & Iris Analysis) ---
        iris_cols = [c for c in df.columns if 'iris' in c.lower()]
        gaze_cols = [c for c in df.columns if 'gaze' in c.lower()]

        if iris_cols or gaze_cols:
            print("      👁️ 检测到眼动数据，启动专项分析...")

            # 1. 视线稳定性 (Gaze Stability)
            if 'gaze_deviation' in df.columns:
                dev_series = df['gaze_deviation'].dropna()
                if len(dev_series) > 0:
                    stability_score = 1.0 / (1.0 + dev_series.std())
                    features['face_gaze_stability_mean'] = float(stability_score)
                    features['face_gaze_deviation_mean'] = float(dev_series.mean())

            # 2. 注视中心度 (Center Fixation) -> Eye Contact
            center_x, center_y = 0.5, 0.5
            dist_list = []

            if 'left_iris_x' in df.columns and 'left_iris_y' in df.columns:
                lx = df['left_iris_x'].dropna()
                ly = df['left_iris_y'].dropna()
                if len(lx) == len(ly) and len(lx) > 0:
                    dist = np.sqrt((lx - center_x) ** 2 + (ly - center_y) ** 2)
                    dist_list.append(dist)

            if 'right_iris_x' in df.columns and 'right_iris_y' in df.columns:
                rx = df['right_iris_x'].dropna()
                ry = df['right_iris_y'].dropna()
                if len(rx) == len(ry) and len(rx) > 0:
                    dist = np.sqrt((rx - center_x) ** 2 + (ry - center_y) ** 2)
                    dist_list.append(dist)

            if dist_list:
                all_dist = pd.concat(dist_list)
                avg_dist = all_dist.mean()
                eye_contact_score = max(0.0, 1.0 - (avg_dist * 2))
                features['face_eye_contact_ratio'] = float(eye_contact_score)
                features['face_gaze_dispersion_mean'] = float(avg_dist)

            # 3. Gaze 方向统计
            for col in gaze_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    series = df[col].dropna()
                    if len(series) > 0:
                        features[f"face_{col}_mean"] = float(series.mean())
                        features[f"face_{col}_std"] = float(series.std())

        if len(features) == 0:
            print("      ⚠️ 未检测到关键词列，执行全量数值扫描...")
            features.update(self._extract_numeric_stats(df, prefix="face_"))

        return features

    # ---------------------------------------------------------
    # 2. 手势特征提取
    # ---------------------------------------------------------
    def _extract_gesture_features(self) -> Optional[Dict[str, Any]]:
        if 'gesture' not in self.data or self.data['gesture'].empty:
            return None

        df = self.data['gesture']
        features = {}
        features.update(self._extract_numeric_stats(df, prefix="gesture_"))

        coord_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                      if any(x in c.lower() for x in ['x_', 'y_', 'z_', 'coord', 'landmark'])]
        if coord_cols:
            features['gesture_global_motion_amplitude'] = float(df[coord_cols].std().sum())
        return features

    # ---------------------------------------------------------
    # 3. 语音特征提取 (终极修复版：强制文本扫描 + 流畅度代理)
    # ---------------------------------------------------------
    def _extract_voice_features(self, modality_key: str) -> Optional[Dict[str, Any]]:
        if modality_key not in self.data or self.data[modality_key].empty:
            return None

        df = self.data[modality_key]
        features = {}
        prefix = "research" if "research" in modality_key else "interview"
        is_small_sample = len(df) < 20

        # A. 情感列
        if 'emotion' in df.columns:
            if pd.api.types.is_numeric_dtype(df['emotion']):
                features.update(self._extract_numeric_stats(df[['emotion']], prefix=f"{prefix}_emotion_"))
            else:
                features[f'{prefix}_dominant_emotion'] = str(df['emotion'].mode()[0]) if not df[
                    'emotion'].empty else "unknown"
                features[f'{prefix}_emotion_diversity'] = float(df['emotion'].nunique())

        # B. 数值列扫描与别名映射
        text_cols_found = []

        for col in df.columns:
            if col == 'emotion' or 'timestamp' in col.lower() or 'id' in col.lower():
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                min_rows = 1000 if is_small_sample else 20
                stats = self._extract_numeric_stats(df[[col]], prefix=f"{prefix}_{col}_", min_rows_for_trend=min_rows)
                features.update(stats)

                col_lower = col.lower()
                base_name = f"{prefix}_{col}"

                # 1. 语调变化映射
                if 'pitch' in col_lower and 'mean' in col_lower:
                    features[f"{prefix}_pitch_variation_mean"] = stats.get(f"{base_name}_mean", 0)

                # 2. 能量映射 (保留微小值)
                if 'energy' in col_lower and 'mean' in col_lower:
                    features[f"{prefix}_energy_level"] = stats.get(f"{base_name}_mean", 0)

                # 3. 停顿映射
                if 'pause' in col_lower and 'duration' in col_lower and 'mean' in col_lower:
                    features[f"{prefix}_pause_duration_mean"] = stats.get(f"{base_name}_mean", 0)

                # 4. 【新增】流畅度代理计算 (如果没有直接分数)
                # 假设：说话占比高 + 停顿短 + 语速适中 = 流畅
                if 'speech_ratio' in col_lower and 'mean' in col_lower:
                    ratio = stats.get(f"{base_name}_mean", 0)
                    # 简单代理：直接用占比作为流畅度基础 (0-1 -> 0-100)
                    features[f"{prefix}_fluency_score_mean"] = ratio * 100
                    features[f"{prefix}_fluency_proxy"] = ratio * 100

            # C. 【核心修复】文本列强制扫描
            elif 'text' in col.lower() or 'content' in col.lower() or 'answer' in col.lower() or 'script' in col.lower():
                text_cols_found.append(col)
                lengths = df[col].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                features[f'{prefix}_{col}_avg_length'] = float(lengths.mean())

                # 强制计算逻辑密度
                if 'text' in col.lower() or 'content' in col.lower() or 'answer' in col.lower():
                    text_concat = " ".join(df[col].dropna().astype(str)).lower()
                    if len(text_concat) > 10:  # 确保有内容
                        logic_keywords = ['因为', '所以', '但是', '然而', '分析', '认为', '假设', '实验', '数据',
                                          '首先', '其次', '最后', '因此', '由于', '尽管', '如果', '那么', '综上所述']
                        hit_count = sum(1 for k in logic_keywords if k in text_concat)
                        density = float(hit_count / max(1, len(text_concat)))

                        # 【关键】写入不带模态前缀的通用键名，确保 Mapper 一定能搜到！
                        # 同时也写入带前缀的
                        features[f'logic_keyword_density'] = density
                        features[f'{prefix}_logic_keyword_density'] = density
                        features[f'{prefix}_{col}_logic_density'] = density

        # D. 【兜底】如果没找到文本列，尝试查找任何包含中文字符的列
        if not text_cols_found:
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().head(1)
                    if not sample.empty and any('\u4e00' <= c <= '\u9fff' for c in str(sample.iloc[0])):
                        # 发现潜在文本列
                        text_concat = " ".join(df[col].dropna().astype(str)).lower()
                        if len(text_concat) > 10:
                            logic_keywords = ['因为', '所以', '但是', '然而', '分析', '认为', '假设', '实验', '数据']
                            hit_count = sum(1 for k in logic_keywords if k in text_concat)
                            density = float(hit_count / max(1, len(text_concat)))
                            features[f'logic_keyword_density'] = density
                            features[f'{prefix}_logic_keyword_density'] = density
                            print(f"      💡 自动发现文本列 '{col}' 并计算逻辑密度。")
                            break

        return features
    def get_summary_report(self, limit_per_modality: int = 20) -> str:
        if not self.features: return "未提取到任何有效特征。"
        lines = ["\n=== 📊 标准化特征量化摘要 ==="]
        lines.append("注：所有键名已规范化，优先展示 Mean/Trend/Freq 指标。")
        lines.append("-" * 70)

        for modality, feats in self.features.items():
            lines.append(f"\n【{modality.upper()}】")

            def sort_key(k):
                if any(x in k for x in ['mean', 'score', 'freq', 'density', 'trend']):
                    return 0
                elif any(x in k for x in ['std', 'sum', 'complexity']):
                    return 1
                else:
                    return 2

            sorted_keys = sorted(feats.keys(), key=sort_key)
            count = 0
            for k in sorted_keys:
                if count >= limit_per_modality:
                    lines.append(f"  ... (还有 {len(sorted_keys) - limit_per_modality} 个隐藏特征)")
                    break
                v = feats[k]
                if isinstance(v, float):
                    if v > 1000 or (v < 0.001 and v > 0):
                        lines.append(f"  - {k}: {v:.4e}")
                    else:
                        lines.append(f"  - {k}: {v:.4f}")
                else:
                    lines.append(f"  - {k}: {v}")
                count += 1
        return "\n".join(lines)


# =========================================================
# 【新增】眼动特征专项检查报告
# =========================================================
def inspect_gaze_features(engine: PsychologicalFeatureEngine):
    if 'face' not in engine.data:
        print("❌ 未找到面部数据。")
        return

    df_face = engine.data['face']
    print("\n" + "=" * 70)
    print("👁️  眼动特征专项深度分析报告")
    print("=" * 70)

    print("\n📊 [原始数据] 检测到的眼动相关列 (CSV):")
    gaze_keywords = ['iris', 'gaze', 'pupil', 'eye']
    original_cols = [c for c in df_face.columns if any(k in c.lower() for k in gaze_keywords)]

    if original_cols:
        print(f"   ✅ 共发现 {len(original_cols)} 列")
        print("   📈 关键列统计预览:")
        key_cols = ['left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y',
                    'gaze_direction_x', 'gaze_direction_y', 'gaze_deviation']

        for col in key_cols:
            if col in df_face.columns:
                series = df_face[col].dropna()
                if len(series) > 0:
                    print(
                        f"      • {col:<20}: 均值={series.mean():.4f}, 标准差={series.std():.4f}, 范围=[{series.min():.3f}, {series.max():.3f}]")
    else:
        print("   ⚠️ 原始数据中未发现明显的眼动列。")

    if not engine.features or 'face' not in engine.features:
        print("\n❌ 特征提取失败。")
        return

    face_feats = engine.features['face']
    print("\n🧠 [提取结果] 生成的眼动相关量化特征:")
    extracted_gaze_keys = [k for k in face_feats.keys() if
                           any(keyword in k for keyword in ['gaze', 'iris', 'eye_contact', 'dispersion', 'stability'])]

    priority_keywords = ['stability', 'contact', 'dispersion', 'deviation']

    def sort_priority(k):
        for i, kw in enumerate(priority_keywords):
            if kw in k: return i
        return 99

    extracted_gaze_keys.sort(key=sort_priority)

    if extracted_gaze_keys:
        print(f"   ✅ 共提取 {len(extracted_gaze_keys)} 个眼动特征：\n")
        print(f"   {'特征键名 (Key)':<45} | {'数值 (Value)':>12} | {'说明'}")
        print("-" * 75)

        for key in extracted_gaze_keys:
            val = face_feats[key]
            desc = ""
            if isinstance(val, float):
                if 'stability' in key:
                    desc = "👁️ 视线稳定性 (越高越专注)"
                elif 'contact' in key:
                    desc = "👁️ 眼神接触比例 (越高越自信)"
                elif 'dispersion' in key:
                    desc = "👁️ 视线离散度 (越低越聚焦)"
                elif 'deviation' in key:
                    desc = "👁️ 视线偏差均值"
                elif 'std' in key:
                    desc = "📉 波动性 (标准差)"
                elif 'mean' in key:
                    desc = "📊 平均水平"

                print(f"   {key:<45} | {val:>12.6f} | {desc}")
            else:
                print(f"   {key:<45} | {str(val):>12} | -")

        print("-" * 75)
        stability_val = face_feats.get('face_gaze_stability_mean', 0)
        contact_val = face_feats.get('face_eye_contact_ratio', 0)

        print("\n💡 智能分析结论:")
        if stability_val > 0.8:
            print("   ✅ 被测者视线非常稳定，显示出极高的专注度。")
        elif stability_val > 0.5:
            print("   ➖ 被测者视线稳定性适中。")
        else:
            print("   ⚠️ 被测者视线波动较大，可能注意力分散或紧张。")

        if contact_val > 0.7:
            print("   ✅ 被测者眼神接触良好，表现出较强的自信心。")
        elif contact_val > 0.4:
            print("   ➖ 被测者眼神接触一般。")
        else:
            print("   ⚠️ 被测者眼神接触较少，可能存在回避或紧张情绪。")
    else:
        print("   ❌ 未提取到预期的核心眼动特征。")
    print("=" * 70)


# --- 本地测试入口 ---
if __name__ == "__main__":
    from data_loader import LogDataLoader

    print("=== 测试 Feature Engine (最终增强版) ===")

    loader = LogDataLoader()
    data = loader.get_fused_latest_data()

    if data:
        engine = PsychologicalFeatureEngine(data)
        features = engine.extract_all_features()

        if features:
            print(engine.get_summary_report(limit_per_modality=10))
            inspect_gaze_features(engine)

            sample_key = list(features['face'].keys())[0]
            if '__' not in sample_key and not sample_key.startswith('_'):
                print(f"\n✅ [验证] 命名规范化成功：示例 '{sample_key}' 格式正确。")
            else:
                print(f"\n⚠️ [验证] 命名存在问题：{sample_key}")

            # 检查语音别名是否生成
            if 'voice_interview' in features:
                vi = features['voice_interview']
                if 'interview_logic_keyword_density' in vi:
                    print("✅ [验证] 语音逻辑密度别名生成成功。")
                if 'interview_pitch_variation_mean' in vi:
                    print("✅ [验证] 语音语调变化别名生成成功。")

        else:
            print("\n⚠️ 未提取到特征。")
    else:
        print("❌ 数据加载失败。")
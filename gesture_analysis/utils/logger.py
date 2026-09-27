"""
日志工具模块

提供手势和肩部分析结果的日志记录功能
支持结构化 CSV 日志，便于后续分析与审计。
"""

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from ..config import LOGS_DIR, LOG_CONFIG

NONE_SESSION = "NONE"          # 无 id 时的显式占位,与另两个 logger 及 asr/session.py 同值(测试守住)


def _is_num(v) -> bool:
    """能不能当有限浮点数用(用于 world 列:能就统一保留 4 位小数)。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


class GestureLogger:
    """手势分析日志记录器"""

    def __init__(self, log_dir: Optional[str] = None, log_file_name: Optional[str] = None,
                 log_file_path: Optional[str] = None, session_id: Optional[str] = None):
        """
        初始化日志记录器

        参数:
            log_dir: 日志目录路径，若为 None 则使用 config.LOGS_DIR
            log_file_name: 日志文件名模板，若为 None 则使用 config 中的模板
            log_file_path: 直接指定日志文件完整路径（提供后忽略 log_dir/log_file_name）
            session_id: 会话ID，用作文件名的一部分（缺省时退回 NONE + 时间戳）
        """
        self.session_id = session_id or NONE_SESSION

        if log_file_path:
            self.log_file = Path(log_file_path)
            self.log_dir = self.log_file.parent
        else:
            self.log_dir = Path(log_dir) if log_dir else Path(LOGS_DIR)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sid_part = self.session_id if session_id else f"{NONE_SESSION}_{timestamp}"
            file_template = log_file_name or LOG_CONFIG['gesture_log_file']
            self.log_file = self.log_dir / file_template.format(timestamp=sid_part)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 定义字段名（包含所有原始特征数据和屏幕显示的角度）
        self.fieldnames = [
            # 会话
            "session_id",               # 会话ID（首列：报告侧按它归堆/排除）

            # 时间戳
            "timestamp",                # Unix 时间戳（秒）
            "timestamp_iso",            # ISO 8601 格式时间（便于阅读）

            # 手部特征
            # ⚠️ 2026-09-27(M3.3 删列):`left_hand_score` / `right_hand_score` 两列已删 ——
            # 理由是「可由同 block 的 jitter 精确重构(`max|diff| = 0`,三场正式素材
            # 1418 行实测)」,零额外信息。**删的是落盘,不是计算**:`resilience_score`
            # 仍是 `HandAnalyzer` 结果字典里的键(情绪推断拿它判有效性)。
            "left_hand_jitter",
            "left_hand_fist_status",
            "left_hand_spread",
            "right_hand_jitter",
            "right_hand_fist_status",
            "right_hand_spread",

            # 手指角度（屏幕显示）
            "left_thumb_angle",
            "left_index_angle",
            "left_middle_angle",
            "left_ring_angle",
            "left_pinky_angle",
            "right_thumb_angle",
            "right_index_angle",
            "right_middle_angle",
            "right_ring_angle",
            "right_pinky_angle",

            # 肩部特征
            # (2026-09-27 M3.3 删列:`shoulder_score` 已删 —— jitter + shrug_level 的确定性
            #  函数;`shoulder_score` 仍是 `ShoulderAnalyzer` 的结果键,情绪推断吃它。)
            "left_shoulder_jitter",
            "right_shoulder_jitter",
            "shrug_level",
            "is_calibrated",

            # 手臂特征
            # (2026-09-27 M3.3 删列:`left/right_arm_score` 与 `left/right_arm_stability` 已删;
            #  `arm_stability` 仍是 `ArmAnalyzer` 的结果键 —— `_compute_arm_score` 的
            #  `stability_bonus` 项吃它,删键会让奖励项**静默**归零。)
            "left_wrist_jitter",
            "left_elbow_jitter",
            "left_arm_angle",
            "right_wrist_jitter",
            "right_elbow_jitter",
            "right_arm_angle",

            # 手臂角度（屏幕显示）
            "left_elbow_angle",
            "right_elbow_angle",
            "left_shoulder_angle",
            "right_shoulder_angle",

            # 上半身特征
            # (2026-09-27 M3.3 删列:`head_score` / `torso_score` / `torso_stability` 三列已删;
            #  `torso_stability` 仍是 `UpperBodyAnalyzer` 的结果键 —— `_compute_torso_score`
            #  吃它。`head_score`/`torso_score` 是分析器自己的输出,留着不占落盘。)
            "head_jitter",
            "head_tilt",
            "torso_jitter",

            # 头部角度（屏幕显示）
            "head_tilt_angle",
            "head_pitch_angle",

            # 肩部角度（屏幕显示）
            "shoulder_angle",

            # 躯干角度（屏幕显示）
            "torso_angle",

            # 情绪特征
            # ⚠️ 2026-09-27(M3.3 删列):`overall_score` 已删 —— 它在三场正式素材上
            # 1418/1418 行**恒 50.0**(情绪推断在活服务里整块没有产出,实测见
            # `l0_columns.json` 的 `deltas[1].reasons[8]`)。下面四列是同一块的其余四列,
            # 在本轮**不动**(判删依据只对 `overall_score` 逐行核过)。
            "emotion_state",
            "feedback",
            "used_features",
            "is_valid",

            # ── 左右手**标签的来源**(2026-09-26 加)────────────────────────────
            # 在此之前 `left_hand_*` / `right_hand_*` 是按**检出顺序**分的(第一只/
            # 第二只),而模型明明给了 handedness。现在按 handedness 分,并把模型
            # 的**原始标签与置信度**落盘 —— 好让"翻没翻对"是可审计的,而不是只能信代码。
            # 空 = 那一槽的左右手**没有依据**(模型没给 handedness 时的兜底),
            # 不是"标签是空字符串"。
            "left_hand_model_label",
            "left_hand_model_label_conf",
            "right_hand_model_label",
            "right_hand_model_label_conf",

            # ── world 米制坐标算出来的抖动/角度(2026-09-26 加)────────────────
            # "取景代理""未除尺度"这些封停理由的根,是上面那些量都在**归一化画面
            # 坐标**上算。这八列用模型的 `pose_world_landmarks`(米制 3D)算**同一套
            # 公式**,所以能与上面那些直接对照,而且与取景无关。
            # 旧列一个不动 —— 换定义会静默改掉报告里所有阈值的含义。
            "left_wrist_jitter_world",
            "left_elbow_jitter_world",
            "right_wrist_jitter_world",
            "right_elbow_jitter_world",
            "left_arm_angle_world",
            "right_arm_angle_world",
            "left_shoulder_jitter_world",
            "right_shoulder_jitter_world",

            # ── 本帧该侧有没有**已署名**的手(2026-09-26 加,Task 6)────────────
            # `1` = 该侧槽**有手**且模型**给了 handedness**;**空** = 不知道。
            # ⚠️ 「有手」与「知道是哪只手」是两件事:端点在没有 handedness 依据时会把一只
            # **来路不明**的手塞进空槽(api/app.py 的兜底支路)—— 那时"有手"是真的、
            # "那是左手"是假的。写 1 就是把不知道的事说成知道。
            # ⚠️ **不写 0**:`0` 的意思是"确定没有这只手",与"没测到"不是一回事。
            # 判据与 `_handedness_cells()` **同一约定、同一入口**(都读 `handedness_info`),
            # 不另起一套;细节差异(它当真值用、本列判 `is not None`)见
            # `_hand_visible_cells()` 的 docstring。
            "hand_visible_left",
            "hand_visible_right",

            # ── 双肩的归一化图像距离(2026-09-26 加,Task 7)──────────────────
            # ★★ **单位是归一化图像单位,不是米。** 转米要相机内参,而本项目**没有任何
            # 内参来源**(2026-09-26 全仓 grep `solvePnP`/`Rodrigues`/`camera_matrix`/
            # 焦距 0 命中,`models/` 下无标定文件)⟹ 米制肩宽只能靠**编一个内参**。
            # 它的用途是**协变量**:报告层拿它作**分母/尺度基准**,消掉取景与距离对别的量的
            # 影响(spec §4.5 规矩)。它测的是**画面尺度**,不是「被测者肩宽多少厘米」。
            # ⚠️ 缺任一肩 / 本帧没有姿态 ⟹ **空**(不写 0 —— 归一化坐标里 0 是「两肩重合」,
            # 一个看着像测量值的假数)。取值口径见 angles.shoulder_width()。
            "shoulder_width"
        ]

        # 写入文件头（仅一次）
        self._write_header()

    # 与 fieldnames 里那八列**同名**:一处声明、一处取值,调用方按列名给
    # (2026-09-26 实测过的坑:上游产出名与下游列名不一致 = 静默的零值)。
    _WORLD_COLUMNS = (
        "left_wrist_jitter_world", "left_elbow_jitter_world",
        "right_wrist_jitter_world", "right_elbow_jitter_world",
        "left_arm_angle_world", "right_arm_angle_world",
        "left_shoulder_jitter_world", "right_shoulder_jitter_world",
    )

    # world 列的落盘小数位(M3.3 B3,2026-09-27:4 → 6)。
    # 为什么改:那 6 个 jitter 列现在是**率**(米/秒),M3 之后量级从 ~0.01 m 掉到
    # ~1e-4…1e-3 m/s —— 4 位小数(步长 1e-4)在那个量级上只剩 **1 位有效数字**,
    # 相关/分位这类下游统计会被量化噪声吃掉(实测:改后 3 场的中位数落在
    # 2e-4…8e-4,4 位小数下相邻两档就能差一倍)。列名/列序一个都没动。
    _WORLD_DECIMALS = 6

    @classmethod
    def _world_cells(cls, world_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        src = world_results or {}
        unknown = sorted(set(src) - set(cls._WORLD_COLUMNS))
        if unknown:
            # 静默丢列正是本项目反复栽的坑 —— 名对不上就先喊出来
            logging.getLogger(__name__).warning(
                "world_results 里有 %d 个名字不在列里,它们不会进日志:%s", len(unknown), unknown)
        out: Dict[str, Any] = {}
        for col in cls._WORLD_COLUMNS:
            v = src.get(col)
            out[col] = "" if v is None else (
                round(float(v), cls._WORLD_DECIMALS) if _is_num(v) else v)
        return out

    @staticmethod
    def _handedness_cells(handedness_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """把 `handedness_info` 摊成四个单元格。

        空字符串是**有意的**:它表示"这一槽的左右手没有依据",与"标签是空串"不是
        一回事;补 0 或补 "Left" 都会把一个未知说成一个已知。
        """
        info = handedness_info or {}
        cells: Dict[str, Any] = {}
        for slot in ("left_hand", "right_hand"):
            entry = info.get(slot)
            label, conf = (entry if entry else ("", ""))
            cells[f"{slot}_model_label"] = label or ""
            cells[f"{slot}_model_label_conf"] = round(float(conf), 3) if conf != "" else ""
        return cells

    @staticmethod
    def _hand_visible_cells(handedness_info: Optional[Dict[str, Any]],
                            hand_present: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """本帧该侧是不是有**已署名**的手 —— 交两个单元格(`hand_visible_left/right`)。

        判据 = **两个条件的合取**,缺一就留空:
          · `handedness_info[槽] is not None` —— **知道**这一槽是哪只手(依据);
          · `hand_present[槽]` —— 这一槽**真的收到了手**(事实)。

        ⚠️ 判据必须是 `handedness_info` 而不是"槽非空":端点的兜底分槽支路
        (`api/app.py` 收到没带 handedness 的手时)会把那只手塞进空槽、并把
        `handedness_info[槽]` 置 `None`。那时"有手"是真的、"那是左手"是假的 ——
        只看"槽非空"就写 1,等于把不知道的事说成知道。

        ⚠️ 没有依据时写 **空串**,**永不写 `0`**:`0` 的意思是"确定没有这只手",
        与"不知道"是两回事(与 `_handedness_cells()` **同一约定、同一入口** —— 都是
        `handedness_info`;不另起一套)。

        ⚠️ **一处与 `_handedness_cells()` 的细微不同,别当成"逐字照抄"**(复核 Minor 3)。
        那个方法把条目当**真值**用(`entry if entry else ("", "")`),本方法判的是
        `is not None`。两处都会出现「label 格空,而本列写 `1`」,但**机制是两个**
        (2026-09-26 复核指出:原文只举了第二种,却按第一种的机制解释 —— 效果描述碰巧对、
        机制说错了;两种机制都由 `tests/test_gesture_task_outputs_wired.py::
        test_the_two_hand_cell_helpers_diverge_on_two_different_mechanisms` 钉住):
          · **甲:假值但非 `None`** 的条目(`()` / `0` / `""` / `{}`)—— 那个三元把**整个
            条目**换成 `("", "")` ⟹ label 与 conf **两格都空**;而本列判 `is not None` ⟹ 写 `1`。
            **`("", 0.9)` 不属于这一类**:它是**非空元组 = 真值**,`_handedness_cells`
            根本不会走 `("", "")` 那一支(原文正是把它当成了本类的例子)。
          · **乙:条目里的 label 元素本身为空**(如 `("", 0.9)`)—— 走**正常分支**,
            label 格空是因为**模型给的那个 label 就是空串**(`label or ""` 只是顺手把
            `None` 之类归一成空串,不是原因);conf 格**有值**(`0.9`)。本列同样写 `1`。
        判据取 `is not None` 是因为表里 `definition` 写的**就是** `handedness_info[槽] is not None`;
        要两边严格同形,得先改表里那句话 —— 那是一次语义选择,不是这里顺手能定的。

        ⚠️ **活路径会不会给出这两种条目**:甲类**活路径产不出** —— 端点在 `entry is not None`
        时写的就是 `(label, conf)` 原样,而 `detectors.py` 的 `handedness` 逐条要么给
        `(category_name, score)` 要么给 `None`,不走三元那条路;乙类**代码上也没被排除**:
        `detectors.py` 取的是 `entry[0].category_name` **原样**,没查过它非空 ⟹ 模型哪天给出
        空 `category_name`,乙类就进得来。2026-09-26 实测:`data/logs/` 下**所有带
        `*_hand_model_label_conf` 列的 gesture 日志**(12 场 / 2933 行 / 5866 格)里,
        乙类的签名(`label == ""` 且 `conf != ""`)的格子 **0 个**。
        ⟹ 准确说法是「**没观测到过**(且甲类今天产不出)」,**不是**「构造上产不出」。

        `hand_present` 缺省(None)= **不知道**,⟹ 两格留空 —— 不拿"没传"当"没有",
        也不拿它当"有"(见 `log()` 的入参说明)。
        """
        info = handedness_info or {}
        present = hand_present or {}
        cells: Dict[str, Any] = {}
        for slot, side in (("left_hand", "left"), ("right_hand", "right")):
            known = info.get(slot) is not None
            cells[f"hand_visible_{side}"] = "1" if (known and present.get(slot)) else ""
        return cells

    def _write_header(self) -> None:
        """写入 CSV 文件头（幂等操作）"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(
        self,
        left_hand_result: Optional[Dict[str, Any]],
        right_hand_result: Optional[Dict[str, Any]],
        shoulder_result: Optional[Dict[str, Any]],
        left_arm_result: Optional[Dict[str, Any]] = None,
        right_arm_result: Optional[Dict[str, Any]] = None,
        upper_body_result: Optional[Dict[str, Any]] = None,
        emotion_result: Optional[Dict[str, Any]] = None,
        angles_data: Optional[Dict[str, Any]] = None,
        handedness_info: Optional[Dict[str, Any]] = None,
        world_results: Optional[Dict[str, Any]] = None,
        hand_present: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        记录分析结果到日志文件

        参数:
            left_hand_result: 左手分析结果
            right_hand_result: 右手分析结果
            shoulder_result: 肩部分析结果
            left_arm_result: 左手臂分析结果
            right_arm_result: 右手臂分析结果
            upper_body_result: 上半身分析结果
            emotion_result: 情绪评估结果
            angles_data: 屏幕显示的所有角度数据
            handedness_info: 每一槽的 `(模型原始标签, 置信度)`;`None`/缺 = 该槽**没有依据**
            hand_present: 每一槽**是否真的收到了手**(键 `left_hand`/`right_hand`)。
                ⚠️ 它由**端点**给出而不是从 `left_hand_result` 推 —— "槽里有手"是端点
                帧循环里的事实(`used_slots`),而 `left_hand_result` 只是它的一个下游代理。
                缺省 `None` = **不知道** ⟹ `hand_visible_*` 两格留空:
                既不当成"没有手"(那是编一个 0),也不当成"有手"(那会把不知道说成知道)。

                ⚠️ **这个入参今天不改变任何一格**(复核 Minor 4,如实说):当前端点
                (`api/app.py`)只在**填槽的同一迭代内**写 `handedness_info[槽]`,所以
                "有依据"⟹"该槽在 `used_slots` 里",`and present.get(slot)` 恒等于左半。
                它买到的是**面向将来**:哪天有人把 `_fresh(...)` 换回 `get_results()`
                (那个改动会同时让"槽里这个值"变成上一帧的旧值),或者有人**直接调** `log()`
                并只给 `handedness_info` —— 那时少了这半就会把"不知道是哪只手"写成
                "这只手可见",而**不会有任何测试变红**(活路径那几条测不到)。
                所以它是**防回归的冗余**,不是当下正确性的来源。

        返回:
            是否成功写入
        """
        try:
            now = datetime.now()
            data = {
                # 会话
                "session_id": self.session_id,

                # 时间戳
                "timestamp": now.timestamp(),
                "timestamp_iso": now.isoformat(),

                # 手部特征(2026-09-27 M3.3:`*_hand_score` 两列已删 —— 见 `fieldnames`)
                "left_hand_jitter": self._safe_get(left_hand_result, 'jitter', 0.0),
                # ★ 2026-09-27(C 档):这一格装的是**无量纲屈曲比**(`HandAnalyzer._flexion_ratio`),
                # 不再是 0/1 判决 —— 表里 `l0_output` 明写「L0 只吐那个比值,门限推给标定集」。
                # 缺省是 `None`(不是 0.0):掌心/掌长取不出来 ⟹ 落**空**,
                # 「没有分母」不许伪装成「屈曲度 = 0」。
                "left_hand_fist_status": self._safe_get(left_hand_result, 'fist_status', None),
                "left_hand_spread": self._safe_get(left_hand_result, 'spread', 0.0),
                "right_hand_jitter": self._safe_get(right_hand_result, 'jitter', 0.0),
                "right_hand_fist_status": self._safe_get(right_hand_result, 'fist_status', None),
                "right_hand_spread": self._safe_get(right_hand_result, 'spread', 0.0),

                # 左右手标签的来源(见字段说明)。`handedness_info[槽]` 是
                # `(模型原始标签, 置信度)`;None/缺 = 那一槽没有依据 ⟹ 留**空**。
                **self._handedness_cells(handedness_info),

                # world 米制版的抖动/角度:没算出来就**留空**(不补 0 —— 0 是个合法
                # 抖动值,补 0 会把"没测到"说成"测到完全静止")。
                **self._world_cells(world_results),

                # 本帧该侧有没有**已署名**的手(见字段说明与 `_hand_visible_cells`)。
                # 与 `_handedness_cells` 同一判据:没有依据 ⟹ 空,不写 0。
                **self._hand_visible_cells(handedness_info, hand_present),

                # 手指角度（屏幕显示）
                "left_thumb_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'thumb'),
                "left_index_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'index'),
                "left_middle_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'middle'),
                "left_ring_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'ring'),
                "left_pinky_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'pinky'),
                "right_thumb_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'thumb'),
                "right_index_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'index'),
                "right_middle_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'middle'),
                "right_ring_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'ring'),
                "right_pinky_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'pinky'),

                # 肩部特征(2026-09-27 M3.3:`shoulder_score` 已删)
                "left_shoulder_jitter": self._safe_get(shoulder_result, 'left_jitter', 0.0),
                "right_shoulder_jitter": self._safe_get(shoulder_result, 'right_jitter', 0.0),
                "shrug_level": self._safe_get(shoulder_result, 'shrug_level', 0.0),
                "is_calibrated": self._safe_int(shoulder_result, 'is_calibrated', False),

                # 手臂特征(2026-09-27 M3.3:`*_arm_score` / `*_arm_stability` 四列已删)
                "left_wrist_jitter": self._safe_get(left_arm_result, 'wrist_jitter', 0.0),
                "left_elbow_jitter": self._safe_get(left_arm_result, 'elbow_jitter', 0.0),
                "left_arm_angle": self._safe_get(left_arm_result, 'arm_angle', 0.0),
                "right_wrist_jitter": self._safe_get(right_arm_result, 'wrist_jitter', 0.0),
                "right_elbow_jitter": self._safe_get(right_arm_result, 'elbow_jitter', 0.0),
                "right_arm_angle": self._safe_get(right_arm_result, 'arm_angle', 0.0),

                # 手臂角度（屏幕显示）
                "left_elbow_angle": self._safe_get(angles_data, 'left_elbow_angle', None),
                "right_elbow_angle": self._safe_get(angles_data, 'right_elbow_angle', None),
                "left_shoulder_angle": self._safe_get(angles_data, 'left_shoulder_angle', None),
                "right_shoulder_angle": self._safe_get(angles_data, 'right_shoulder_angle', None),

                # 上半身特征(2026-09-27 M3.3:`head_score` / `torso_score` / `torso_stability` 已删)
                "head_jitter": self._safe_get(upper_body_result, 'head_jitter', 0.0),
                "head_tilt": self._safe_get(upper_body_result, 'head_tilt', 0.0),
                "torso_jitter": self._safe_get(upper_body_result, 'torso_jitter', 0.0),

                # 头部角度（屏幕显示）
                "head_tilt_angle": self._safe_get(angles_data, 'head_tilt_angle', None),
                "head_pitch_angle": self._safe_get(angles_data, 'head_pitch_angle', None),

                # 肩部角度（屏幕显示）
                "shoulder_angle": self._safe_get(angles_data, 'shoulder_angle', None),

                # 躯干角度（屏幕显示）
                "torso_angle": self._safe_get(angles_data, 'torso_angle', None),

                # 双肩的归一化图像距离(**协变量**,不是米 —— 见字段说明)。
                # 缺 ⟹ 空:`_safe_get` 在"端点没放这个键"时交 `None`,而 **csv 把 `None`
                # 落成空串**(与左邻那 18 个角度列同一机制 —— 它们也走 `default=None`,
                # 而 `left_thumb_angle == ""` 已被 tests/test_gesture_task_outputs_wired.py 钉住)。
                "shoulder_width": self._safe_get(angles_data, 'shoulder_width', None),

                # 情绪特征(2026-09-27 M3.3:`overall_score` 已删 —— 恒 50.0,零信息)
                "emotion_state": self._safe_get(emotion_result, 'emotion_state', ''),
                "feedback": self._safe_get(emotion_result, 'feedback', ''),
                "used_features": self._safe_get(emotion_result, 'used_features', 'none'),
                "is_valid": bool(self._safe_get(emotion_result, 'is_valid', False))
            }

            with open(self.log_file, 'a', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(data)
            return True

        except Exception as e:
            print(f"❌ 日志记录失败: {str(e)}")
            return False

    @staticmethod
    def _safe_get(obj: Optional[Dict], key: str, default):
        """安全获取字典值。

        ⚠️ **`obj is None` ⟹ 返回空字符串,不是 default**(2026-09-26 改)。
        `None` 在这里的意思是"本帧这一路没有数据"(端点对没喂过的分析器交 None):
        补 `default`(0.0/50.0)会把"没测到"写成一个看着合法的测量值;
        交 `default` 之前的老行为还有第二层错 —— 交的是**上一帧的旧值**
        (分析器保留上次结果),而"上一帧的度量"顶替"这一帧的度量"不留痕迹。
        """
        if obj is None:
            return ""
        if not isinstance(obj, dict):
            return default
        return obj.get(key, default)

    @staticmethod
    def _safe_int(obj: Optional[Dict], key: str, default):
        """`_safe_get` 的整数版:空 ⟹ 空(端点是 None 时不能 `int("")`)。"""
        v = GestureLogger._safe_get(obj, key, default)
        if v == "":
            return ""
        try:
            return int(v)
        except (TypeError, ValueError):
            return int(default)

    def _safe_get_angle(self, angles_data: Optional[Dict], finger_angles_key: str, finger_name: str):
        """安全获取手指角度数据"""
        if not isinstance(angles_data, dict):
            return None
        finger_angles = angles_data.get(finger_angles_key)
        if not isinstance(finger_angles, dict):
            return None
        angle = finger_angles.get(finger_name)
        return round(angle, 1) if angle is not None else None

    def get_log_path(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)
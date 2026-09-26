"""手指角度与姿态角度 —— 从 `examples/` 移植进**活路径**(2026-09-26)。

为什么要有这个文件:gesture 日志里那 18 列(10 个手指角度 +
`left/right_elbow_angle` / `left/right_shoulder_angle` / `head_tilt_angle` /
`head_pitch_angle` / `shoulder_angle` / `torso_angle`)**此前从来没有产出方** ——
端点调 `logger.log()` 时压根没传 `angles_data`,`_safe_get_angle` 于是永远拿到 None,
整列全空(实测该场 275 帧全空)。而算这些角度的代码**只活在
`gesture_analysis/examples/run_realtime_analyzer.py` 里**。

定义**照实移植,不另起一套** —— 两处各存一份定义迟早漂移(§4.10 那条教训):

| 量 | 定义(examples 原样) |
|---|---|
| 关节角 | ∠(a, b, c),**b 为顶点** |
| 可见度门限 | **0.6**:低于它的点不参与,结果写 `None`(空)—— **不补 0** |
| `left/right_elbow_angle` | ∠(肩, 肘, 腕) |
| `left/right_shoulder_angle` | ∠(髋, 肩, 肘) |
| `shoulder_angle` | 左肩连线与水平线的夹角:∠((左肩x−0.1, 左肩y), 左肩, 右肩) |
| `head_tilt_angle` | 双耳连线与水平线的夹角:∠((左耳x−0.1, 左耳y), 左耳, 右耳) |
| `head_pitch_angle` | `abs(90 − ∠((鼻x, 鼻y−0.1), 鼻, 双耳中点))` |
| `torso_angle` | `abs(90 − ∠((肩中点x, 肩中点y−0.1), 肩中点, 髋中点))` |
| 手指角度 | 每指取三个关节(mp 手部索引):拇指 2-3-4、食指 5-6-8、中指 9-10-12、无名指 13-14-16、小指 17-18-20 |

`dims=3` 走 x/y/z(现成的 **world 米制坐标**用它):角度与尺度无关,但**深度**该算进去 ——
臂朝镜头伸出去时,只看 x/y 会把一个 3D 角读小。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

# 可见度门限:与 examples 一致(0.6)。
VISIBILITY_FLOOR = 0.6

# mediapipe 姿态的 33 点里我们用到的那几个(名字 → 下标,按官方拓扑)。
_POSE = {
    "nose": 0,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
}

# 手指 → 三个关节的索引(与 examples/main_integrator.py 一致)。
_FINGER_JOINTS = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


def _vec(point, dims: int) -> tuple:
    """从 landmark 对象或 dict 里取坐标。

    `dims=3` 时**必须有 z**:没有就退回 2 维 —— 给一个缺失的 z 补 0,
    等于把"点在镜头平面上"当成事实,3D 角会被算小。
    """
    if isinstance(point, dict):
        get = point.get
    else:
        get = lambda k, d=None: getattr(point, k, d)      # noqa: E731
    x, y = float(get("x")), float(get("y"))
    if dims == 3:
        z = get("z")
        if z is None:
            return (x, y)
        return (x, y, float(z))
    return (x, y)


def _visibility(point) -> float:
    if isinstance(point, dict):
        return float(point.get("visibility", 1.0))
    return float(getattr(point, "visibility", 1.0))


def joint_angle(a, b, c, dims: int = 2) -> Optional[float]:
    """∠(a, b, c),b 为顶点,返回**度**。三点任一是空 / 退化 ⟹ None。"""
    if a is None or b is None or c is None:
        return None
    try:
        va, vb, vc = _vec(a, dims), _vec(b, dims), _vec(c, dims)
    except (TypeError, ValueError):
        return None
    dims = min(len(va), len(vb), len(vc))      # 有哪个点没 z 就整体退回它在的那一维
    if dims < 2:
        return None
    ba = [va[i] - vb[i] for i in range(dims)]
    bc = [vc[i] - vb[i] for i in range(dims)]
    na = math.sqrt(sum(x * x for x in ba))
    nb = math.sqrt(sum(x * x for x in bc))
    if na == 0 or nb == 0:
        return None
    cos = sum(ba[i] * bc[i] for i in range(dims)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def finger_angles(hand_landmarks, dims: int = 2) -> Dict[str, float]:
    """一只手 → `{拇指: 度, …}`。点数不够就交**空字典**(不编 0)。"""
    if not hand_landmarks or len(hand_landmarks) < 21:
        return {}
    out: Dict[str, float] = {}
    for finger, (i, j, k) in _FINGER_JOINTS.items():
        deg = joint_angle(hand_landmarks[i], hand_landmarks[j], hand_landmarks[k], dims)
        if deg is not None:
            out[finger] = round(deg, 2)
    return out


def pose_angles(pose_landmarks, dims: int = 2) -> Dict[str, float]:
    """33 个姿态点 → 那 8 个角度键。可见度不足 / 点数不够 ⟹ 该键**缺席**(不补 0)。"""
    if not pose_landmarks or len(pose_landmarks) < 33:
        return {}

    pt = {name: pose_landmarks[idx] for name, idx in _POSE.items()}
    out: Dict[str, float] = {}

    def ok(*names) -> bool:
        return all(_visibility(pt[n]) > VISIBILITY_FLOOR for n in names)

    def put(key: str, deg: Optional[float]):
        if deg is not None:
            out[key] = round(deg, 2)

    if ok("left_shoulder", "left_elbow", "left_wrist"):
        put("left_elbow_angle", joint_angle(pt["left_shoulder"], pt["left_elbow"],
                                            pt["left_wrist"], dims))
    if ok("right_shoulder", "right_elbow", "right_wrist"):
        put("right_elbow_angle", joint_angle(pt["right_shoulder"], pt["right_elbow"],
                                             pt["right_wrist"], dims))
    if ok("left_hip", "left_shoulder", "left_elbow"):
        put("left_shoulder_angle", joint_angle(pt["left_hip"], pt["left_shoulder"],
                                               pt["left_elbow"], dims))
    if ok("right_hip", "right_shoulder", "right_elbow"):
        put("right_shoulder_angle", joint_angle(pt["right_hip"], pt["right_shoulder"],
                                                pt["right_elbow"], dims))

    # ⚠️ 下面四个是**画面内的倾斜角**(线与水平/垂直的夹角),**固定用 2D** ——
    #    它们本来就是图像平面上的概念,喂 3D 没有定义。dims 只管上面那几个关节角。
    if ok("left_shoulder", "right_shoulder"):
        ls = _vec(pt["left_shoulder"], 2)
        put("shoulder_angle", joint_angle({"x": ls[0] - 0.1, "y": ls[1]},
                                          pt["left_shoulder"], pt["right_shoulder"], 2))

    if ok("left_ear", "right_ear"):
        le = _vec(pt["left_ear"], 2)
        put("head_tilt_angle", joint_angle({"x": le[0] - 0.1, "y": le[1]},
                                           pt["left_ear"], pt["right_ear"], 2))

    if ok("nose", "left_ear", "right_ear"):
        le = _vec(pt["left_ear"], 2)
        re_ = _vec(pt["right_ear"], 2)
        no = _vec(pt["nose"], 2)
        ear_center = {"x": (le[0] + re_[0]) / 2, "y": (le[1] + re_[1]) / 2}
        deg = joint_angle({"x": no[0], "y": no[1] - 0.1}, pt["nose"], ear_center, 2)
        if deg is not None:
            put("head_pitch_angle", abs(90 - deg))

    if ok("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        ls, rs = _vec(pt["left_shoulder"], 2), _vec(pt["right_shoulder"], 2)
        lh, rh = _vec(pt["left_hip"], 2), _vec(pt["right_hip"], 2)
        sc = {"x": (ls[0] + rs[0]) / 2, "y": (ls[1] + rs[1]) / 2}
        hc = {"x": (lh[0] + rh[0]) / 2, "y": (lh[1] + rh[1]) / 2}
        deg = joint_angle({"x": sc["x"], "y": sc["y"] - 0.1}, sc, hc, 2)
        if deg is not None:
            put("torso_angle", abs(90 - deg))

    return out

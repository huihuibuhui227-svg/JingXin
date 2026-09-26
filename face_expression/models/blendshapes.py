"""FaceLandmarker 的 blendshape 名字表 —— **单一真源**(探测器与日志都从这里取)。

2026-09-26 实测:`face_landmarker.task` 的 blendshape 头输出 **52** 个分数
(ARKit 风格:`eyeBlinkLeft` / `jawOpen` / `browDownLeft` / `mouthSmileRight` …),
而在这之前 `FaceLandmarkerOptions` **没开** `output_face_blendshapes` ⟹ 这 52 个数
一个都没进过日志(全仓 `grep blendshape` 零命中)。

为什么它值得进日志:`au_calculator.py` 里那些 `au*` 列是**手写几何比率**
(实测 `au7_eye_squeeze` 229/229 帧恰好 = `1 - avg_ear`,零额外信息);
blendshape 是**模型训练出来的**输出,是有依据得多的 AU 近似
(如 `eyeBlinkLeft/Right` 直接就是眨眼、`jawOpen` 直接就是张口)。
两者**并存**:旧列不动(M3 逐列重构时再对照决定谁留下),新列加在后面。

⚠️ 名字是**从模型实际输出取的**(用留存真帧跑出来的),不是照文档抄的。
换模型/换 mediapipe 版本时,`unknown_names()` 会把多出来的名字报出来 ——
静默丢列正是本项目反复栽的那个坑(§4.10)。
"""

# 列名前缀:与手写比率的 `au*` 明确区分开,一眼看得出这一列是模型给的。
BS_COLUMN_PREFIX = "bs_"

# 顺序取自模型实际输出顺序。`_neutral` 是模型自带的"中性"类(不是肌肉动作),
# 一并留下 —— 它是模型的输出,替模型决定"哪个不算"是另一个决定。
BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)


def blend_column(name: str) -> str:
    """blendshape 名 → 日志列名。列名只在这里生成,别在调用处拼字符串。"""
    return f"{BS_COLUMN_PREFIX}{name}"


BLENDSHAPE_COLUMNS: tuple[str, ...] = tuple(blend_column(n) for n in BLENDSHAPE_NAMES)


def unknown_names(seen) -> list[str]:
    """本次识别里出现了、但不在表里的名字 —— 调用方负责把它**报出来**,不许静默丢。

    换模型版本时这条会先响,而不是等到"某一列永远是空的"才发现。
    """
    return sorted(set(seen) - set(BLENDSHAPE_NAMES))

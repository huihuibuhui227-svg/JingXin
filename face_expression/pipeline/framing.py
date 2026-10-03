"""帧取景阶梯：整幅画面检不到人脸时，逐步裁切放大再试。

## 为什么需要（2026-10-03 实测确定）

23/44 场面部检出率是 **0.0%**。查下来的根因**不是**代码坏、也**不是**画面质量差：

- 肉眼确认留存帧里人脸清清楚楚（1280×720、正面、均值亮度 88.3）；
- 把留存帧喂回**同一个** `VideoPipeline` 重跑，未裁切 **0/60**；
- **把画面中央裁掉一半（等于放大 2 倍）⟹ 60/60（100%）**。

也就是说 mediapipe 的**正面人脸检测器对"脸占画面的比例"敏感**：受试者坐得离摄像头远、
脸在画面里占比小时就检不到。

> 这同时解释了「为什么手势 100% 而面部 0%」：手势走 pose/hand 检测器，在**半身尺度**
> 上工作、对距离不敏感；面部走正面人脸检测器，对脸的大小敏感。同一个机位手能测、
> 脸测不到，是正常的。

## ⚠️ 为什么**不能**无脑全局裁切

实测：同一帧、两路都检出时对比 190 列特征，**142 列会变**。原因有两类：

1. `bs_*`（52 个 blendshape）是**模型直接输出**，对输入分辨率敏感；
2. `*_trend` / `*_volatility` / `is_blink` 是**滚动窗口**或**阈值判定**算的，
   单帧微扰会传播、放大。

而报告层**确实消费** `_trend`（`report_frontend/` 有引用）。所以对**本来就能检出**的
场次做裁切是**净损害**，不是优化。

## 所以本模块的取向

1. **只在全画幅失败时才裁** —— 对已正常的场次行为逐字节不变；
2. **取景在整场内固定**（`probe_crop` 按场次定一次）—— 逐帧变取景会污染
   `*_trend`，那比任何一种单一取景都坏；
3. **判据是"检出/未检出"，不是"特征值好看"**。本项目杀的就是"把没测到看成测到了"，
   所以判据只有一个：`emotion != 'no_face'`。

> 踩过的坑：第一次重跑时我用"特征字典非空"当检出判据，于是 `emotion='no_face'`
> 也被算成检出，得出**假的 100%**。别再用那个判据 —— 用 `is_detected()`。

## 它**不**解决什么

裁切只让**检测器找到脸**。它不会让特征更准 —— 恰恰相反，如上面的实测，
裁切会让一部分派生列漂移。所以对"0% 的场次"它是**从无到有**（任何值都好过没有），
对"已正常的场次"它是**从有到偏**（所以不做）。
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

#: 取景阶梯：(名字, 裁切框 | None)。裁切框是 (x0, y0, x1, y1) 的**画面比例**。
#: 第一级必须是 `None`（全画幅）—— 那是既有行为，能检出就不许动。
CROP_LADDER: Tuple[Tuple[str, Optional[Tuple[float, float, float, float]]], ...] = (
    ("full", None),
    # 中央一半（放大 2 倍）。实测在 0% 的场次上从 0/60 变 60/60。
    ("center50", (0.25, 0.00, 0.75, 0.75)),
    ("center40", (0.30, 0.05, 0.70, 0.65)),
    ("center30", (0.35, 0.10, 0.65, 0.55)),
)

#: 探针帧数：按场次定取景时，每级试多少帧。
PROBE_FRAMES = 12

#: 探针通过线：这一级至少这么多帧检出才算可用。
#: 取 1/3 而不是 1 —— 只中 1 帧的取景在整场上多半还是废的。
PROBE_MIN_HIT_RATIO = 1.0 / 3.0


def is_detected(feats: Optional[dict]) -> bool:
    """★ 全仓**唯一**该用的检出判据。

    为什么不能用「特征字典非空」：未检出时 `VideoPipeline` 也会给出一个非空字典
    （`{'emotion': 'no_face', 'timestamp': …}`）。用非空判会把"没测到"算成"测到了"，
    正是本项目在杀的形态（实测踩过，得出过假的 100%）。
    """
    return bool(feats) and feats.get("emotion") != "no_face"


def apply_crop(image: Any, crop: Optional[Sequence[float]]) -> Any:
    """按比例裁切。`crop=None` 原样返回（保证全画幅是真正的既有行为）。"""
    if crop is None:
        return image
    h, w = image.shape[:2]
    x0, y0, x1, y1 = crop
    # 用 int() 截断而不是 round()：边界只许往里收，不许越界。
    return image[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def probe_crop(
    frames: Sequence[Any],
    timestamps: Sequence[int],
    *,
    ladder: Iterable = CROP_LADDER,
    probe_n: int = PROBE_FRAMES,
    min_hit_ratio: float = PROBE_MIN_HIT_RATIO,
    pipeline_factory=None,
) -> Tuple[str, Optional[Tuple[float, float, float, float]], dict]:
    """把整条阶梯都试一遍，返回**检出最多**的那一级的 `(名字, 裁切框, 统计)`。

    ⚠️ 为什么是"试完取最好"而不是"遇到第一个及格的就停":实测姬欣怡那场
    `center50` 在 12 帧探针上够格，但整场重放只有 79% —— 只试到第一级就收手，
    会错过更合适的那一级。探针只有十几帧，多试几级的代价可以忽略。

    每一级用**独立的 pipeline**（`VideoPipeline` 有跨帧状态：VIDEO 模式要求时间戳
    严格递增，还会算滚动统计），所以不能拿同一个实例连着试几级。

    `frames` 是**已解码的 BGR** 图像（调用方负责读盘，读盘失败的不该混进来）。
    全阶梯都不行时返回 `("none", None, …)` —— 调用方据此保持"没检出"，不要硬凑。
    """
    n = min(probe_n, len(frames))
    if n == 0:
        return "none", None, {"reason": "没有可用的帧", "tried": []}

    if pipeline_factory is None:                      # 延迟导入:模块本身不依赖 cv2/mediapipe
        from face_expression.pipeline.video_pipeline import VideoPipeline

        def pipeline_factory(tag: str):
            return VideoPipeline(session_id=f"probe_{tag}")

    tried = []
    best = (0, "none", None)          # (hits, name, crop)
    need = max(1, int(n * min_hit_ratio))

    for name, crop in ladder:
        p = pipeline_factory(name)
        hits = 0
        for i in range(n):
            img = apply_crop(frames[i], crop)
            try:
                feats = p.process_frame(img, int(timestamps[i]))[2] or {}
            except Exception as e:                     # 单帧异常不该葬送整级探测
                tried.append({"crop": name, "frame": i, "error": f"{type(e).__name__}: {e}"})
                continue
            if is_detected(feats):
                hits += 1
        tried.append({"crop": name, "probed": n, "hits": hits, "need": need})
        # ★ 只在 `hits > best[0]` 时替换 ⟹ 平手时保留**更不激进**的那一级
        #   （阶梯按从宽到窄排，先用上的就是更宽的）。
        if hits > best[0]:
            best = (hits, name, crop)

    if best[0] >= need:
        return best[1], best[2], {"probed": n, "hits": best[0], "tried": tried}

    return "none", None, {"probed": n, "tried": tried}

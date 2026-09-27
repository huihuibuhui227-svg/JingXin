"""重取景归因探针 —— 手势 ② 的两条补做实验(M3.5)。

要证的是什么(表里那 10 个画面坐标 jitter 行的 ② / 6 个 world 行的 ③):
「真像素上不可达,归因给**模型不尺度等变**」。复核认为**方向对但证据不足** ——
原来的实验只有**一条**腿(裁 80% 再放大),而它同时改了**三件事**:
  ① 被测者在画幅里的**相对尺度**(×1.25)—— 这是"尺度"那一项;
  ② **有效分辨率**(裁出 80% 的像素再插值放大 1.25 倍);
  ③ 外侧 20% 的像素**被丢掉**(部位可能出画)。
一条腿动三件事 ⟹ 观察到的变化**归给谁都行**。

本探针把这三件事**拆成三条腿**,每条腿只动一件:

| 腿 | 做什么 | 被测者相对尺度 | 丢像素 | 重采样模糊 |
|---|---|---|---|---|
| `orig` | 原帧(基准) | 1 | 无 | 无 |
| `zoom2` | 整帧 ×2(**不裁不丢**) | 1(**不变**) | 无 | 有(放大插值) |
| `pad` | 原帧放进 1.25× 黑画布居中(不裁不丢) | **×0.8** | 无 | 有(模型那边重新缩放) |
| `blur` | 整帧先 ×0.8 再放回原尺寸(不裁不丢) | 1(**不变**) | 无 | **有**(与 `cropzoom` 同一串插值) |
| `cropzoom` | 中央 80% 裁出再放回原尺寸(复现原实验) | **×1.25** | **有**(外侧 20%) | 有 |

读法:
  · `zoom2` 动 ⟹ 模型对**全局缩放本身**就不等变(与构图无关)。
  · `blur` 动 ⟹ **有效分辨率**这一项单独就能推动它。
  · `pad` 动而 `blur` 不动 ⟹ 推得动它的是**相对尺度**(构图),不是分辨率;
    `cropzoom` 与 `blur` 的差(同一串插值,只差"裁不裁")同样指向这一半。
  · `orig2`(同一帧原样再跑一遍)是**噪声地板**:它若逐格相同 ⟹ 上面的差都是扰动造成的,
    不是跑一次的随机性。

★ 纪律 3:时间戳一律喂**留存账本里的 `declared_ts`**(`replay_retained.load_frames`)。
★ 纪律 1:走**真产出方** —— `gesture_app.process_frame` + 真 `GestureLogger`
  (与 `experiments/replay_retained.py --modality gesture` 同一条腿,不自抄一遍接线)。

用法(WSL 侧):
    PY=~/miniconda3/envs/jingxin/bin/python
    $PY experiments/gesture_reframe_probe.py --session-id 20260926_153202_2b11 \\
        --out-dir ~/shared/jingxin_recordings/20260926_153202_2b11/reframe
    # 产出 <out-dir>/reframe_<腿>.csv,然后分析(便宜、可反复跑):
    $PY experiments/gesture_reframe_probe.py --mode analyze \\
        --out-dir ~/shared/jingxin_recordings/20260926_153202_2b11/reframe
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2                                        # noqa: E402
import numpy as np                                # noqa: E402

from experiments.replay_retained import load_frames   # noqa: E402

# ── 三条腿的名字(顺序 = 报告里的顺序)───────────────────────────────────────
ARMS = ("orig", "zoom2", "pad", "blur", "cropzoom")
# `orig2` 只在有需要时加(噪声地板),它不改变任何东西
NOISE_ARM = "orig2"

PAD_SCALE = 1.25      # 画布放大到 1.25× ⟹ 被测者相对尺度 ×1/1.25 = ×0.8
CROP_FRAC = 0.80      # 原实验:裁中央 80% 再放大回原尺寸 ⟹ 相对尺度 ×1.25

# 被检列(表里那 10 个画面 jitter 行 + 6 个 world 行)—— 只在分析里用到,
# 名字与 `GestureLogger.fieldnames` 逐字一致(两处不同名 = 静默的零值)。
SCREEN_JITTER = ("left/right_hand_jitter, left/right_shoulder_jitter, "
                 "left/right_wrist_jitter, left/right_elbow_jitter, head_jitter, torso_jitter")
WORLD_JITTER = ("left/right_wrist_jitter_world, left/right_elbow_jitter_world, "
                "left/right_shoulder_jitter_world")


def make_variant(img: np.ndarray, arm: str) -> np.ndarray:
    """按腿名造一帧的输入。**每一腿都只动一件**(见模块 docstring 的表)。"""
    h, w = img.shape[:2]
    if arm == "orig" or arm == NOISE_ARM:
        return img
    if arm == "zoom2":
        # 整帧 ×2:不裁、不丢像素。相对几何**一个像素都不变**
        return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    if arm == "pad":
        # 原帧居中放进一张 1.25× 的**黑**画布:不裁、不丢像素,但被测者**相对变小**
        ch, cw = int(round(h * PAD_SCALE)), int(round(w * PAD_SCALE))
        canvas = np.zeros((ch, cw, img.shape[2]), dtype=img.dtype)
        y, x = (ch - h) // 2, (cw - w) // 2
        canvas[y:y + h, x:x + w] = img
        return canvas
    if arm == "blur":
        # 先 ×0.8 再放回原尺寸:不裁、不丢像素,构图**逐像素不变**,
        # 只把有效分辨率降到与 `cropzoom` 同一档(同一串插值)
        small = cv2.resize(img, (int(round(w * CROP_FRAC)), int(round(h * CROP_FRAC))),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if arm == "cropzoom":
        # 原实验:裁中央 80% 再放大回原尺寸(丢掉外侧 20% 的像素)
        x0, y0 = int(round(w * (1 - CROP_FRAC) / 2)), int(round(h * (1 - CROP_FRAC) / 2))
        cw, ch = int(round(w * CROP_FRAC)), int(round(h * CROP_FRAC))
        crop = img[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"未知的腿:{arm}")


def produce(session_dir: Path, out_dir: Path, arms, sid: str) -> int:
    """把一个会话的留存帧按每一腿重放一遍,每腿写一份 CSV(走真 logger)。"""
    import importlib
    import tempfile

    # 见 `replay_retained.replay_gesture` 的说明:`importlib` 取的是**模块本体**,
    # 而 `import gesture_analysis.api.app as ...` 取到的是包 `__init__` 里的 FastAPI 实例。
    gesture_app = importlib.import_module("gesture_analysis.api.app")
    glog = importlib.import_module("gesture_analysis.utils.logger")

    frames, report = load_frames(session_dir, "gesture")
    print(f"[素材] 盘上 {report['frames_on_disk']} 帧,其中 {report['frames_with_ts']} 帧有时间戳;"
          f"账本碎行 {report['torn_lines']} 行")
    if report["missing_ts"]:
        print(f"⚠️  {len(report['missing_ts'])} 帧**没有可用时间戳**,本次不重放;"
              f"前几个:{report['missing_ts'][:5]}", file=sys.stderr)
    if not frames:
        return 0

    imgs = []
    for path, ts in frames:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[跳过] 读不出:{path}", file=sys.stderr)
            continue
        imgs.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ts))

    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        # 每一腿**自己的**分析器与探测器:分析器带 30 帧历史、mediapipe 的 VIDEO 模式
        # 带跟踪状态 ⟹ 两腿共用一个实例就不是"同一模型看两种输入"了,是两种状态。
        key = f"{sid}#{arm}"
        with tempfile.TemporaryDirectory(prefix="reframe_") as junk:
            old_logs_dir = gesture_app.LOGS_DIR
            gesture_app.LOGS_DIR = Path(junk)
            try:
                analyzers = gesture_app.get_or_create_analyzers(key)
                dets = gesture_app.get_or_create_detectors(key)
            finally:
                gesture_app.LOGS_DIR = old_logs_dir
            gesture_app.session_loggers.clear()

            out_csv = out_dir / f"reframe_{arm}.csv"
            if out_csv.exists():
                out_csv.unlink()      # logger 只在文件不存在时写表头 ⟹ 不删会追加
            logger = glog.GestureLogger(log_file_path=str(out_csv), session_id=sid)

            for rgb, ts in imgs:
                frame = gesture_app.process_frame(make_variant(rgb, arm), ts, analyzers, dets)
                logger.log(**frame["log_kwargs"])
            for d in dets.values():
                d.close()
        print(f"[完成] {arm}:{len(imgs)} 帧 -> {out_csv}")
    return len(imgs)


# ══════════════════════════════════════════════════════════════════════════════
# 分析(便宜、可反复跑;**不碰 mediapipe**)
# ══════════════════════════════════════════════════════════════════════════════

def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def paired_ratios(a_rows, b_rows, column: str) -> list[float]:
    """逐帧配对(按行号)的 `b/a`;两格都有数才算一对。"""
    out = []
    for ra, rb in zip(a_rows, b_rows):
        va, vb = _num(ra.get(column, "")), _num(rb.get(column, ""))
        if va is None or vb is None or va == 0.0:
            continue
        out.append(vb / va)
    return out


def quantiles(xs, qs=(0.10, 0.50, 0.90)) -> list[float] | None:
    if not xs:
        return None
    s = sorted(xs)
    out = []
    for q in qs:
        i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        out.append(s[i])
    return out


def spearman(xs, ys) -> float | None:
    """秩相关(不假设线性)—— 用来问「这一列是不是跟着输入尺度一起动」。"""
    if len(xs) < 4:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return None if den == 0 else num / den


def _scale_keys(rows: list[dict]) -> set[str]:
    """CSV 里真的出现的 jitter 列(= 本表要判的那几行的名字)。

    ⚠️ 含 `left/right_hand_jitter`(那两行的值只在检到手的那几十帧上有)—— 排除它们
    会让「10 个画面坐标 jitter 行」少算两行,而那两行正是 §4.2 点名要判的。
    配对数太少的那些列由 `verdict()` 自己交 `insufficient`,不在这里悄悄扔掉。
    """
    if not rows:
        return set()
    return {k for k in rows[0]
            if (k.endswith("_jitter") or k.endswith("_jitter_world"))}


def analyze(out_dir: Path, ref: str = "orig") -> dict:
    """把每一腿与 `ref` 逐列配对,交「这一列变了多少」与「输入自己变了多少」。"""
    ref_rows = _read(out_dir / f"reframe_{ref}.csv")
    cols = sorted(_scale_keys(ref_rows))
    world = [c for c in cols if c.endswith("_jitter_world")]
    screen = [c for c in cols if not c.endswith("_jitter_world")]
    result: dict = {"reference": ref, "arms": {}}
    for arm_csv in sorted(out_dir.glob("reframe_*.csv")):
        arm = arm_csv.stem.replace("reframe_", "")
        if arm == ref:
            continue
        rows = _read(arm_csv)
        n = min(len(rows), len(ref_rows))
        rows, rrows = rows[:n], ref_rows[:n]
        # ★ 尺度比 r_t:模型自己把「被测者有多大」估成了多少(逐帧)
        r_t = paired_ratios(rrows, rows, "shoulder_width")
        entry = {
            "frames_paired": n,
            "r_t": quantiles(r_t),
            "r_t_n": len(r_t),
            "columns": {},
        }
        for col in cols:
            # ⚠️ 配对要**逐帧对齐**:某一列在有些帧是空的(没检到手/没姿态),那时 r_t
            #    也必须取同一批帧 —— 否则算出来的是两条不同长度序列的秩相关,ρ 恒 None。
            pairs, scale = [], []
            for ra, rb in zip(rrows, rows):
                va, vb = _num(ra.get(col, "")), _num(rb.get(col, ""))
                if va is None or vb is None or va == 0.0:
                    continue
                sw_a = _num(ra.get("shoulder_width", ""))
                sw_b = _num(rb.get("shoulder_width", ""))
                if sw_a is None or sw_b is None or sw_a == 0.0:
                    continue
                pairs.append(vb / va)
                scale.append(sw_b / sw_a)
            q = quantiles(pairs)
            entry["columns"][col] = {
                "n": len(pairs),
                "q": q,
                "median": (statistics.median(pairs) if pairs else None),
                "rho_with_r_t": spearman(pairs, scale),
                "_pairs": pairs,          # 只在本函数内用(判据的输入),不进 JSON
            }
        control_pairs: list[float] = []
        for label, group in (("world_control", world), ("screen_normalized", screen)):
            devs = []
            for c in group:
                info = entry["columns"].get(c) or {}
                if label == "world_control":
                    control_pairs.extend(info.get("_pairs") or [])
                # ⚠️ 中位比可以是 **0**(手部那两列在热身帧上两边都是 0 ⟹ 比值 0):
                #    这种列在"变了多少"这个口径下**不可测**(0 在乘性口径里不是一个数),
                #    跳过它并计入 `unmeasurable`,不要让它把整条腿的汇总打崩。
                if info.get("q") and info["q"][1] > 0:
                    devs.append(abs(math.log(info["q"][1])))
                elif info.get("q"):
                    entry.setdefault("unmeasurable", []).append(c)
            q = quantiles(devs)
            entry[label] = {
                "columns": len(devs),
                "median_abs_log_ratio": (statistics.median(devs) if devs else None),
                "p90_abs_log_ratio": (q[2] if q else None),
                "max_median_abs_log": (max(devs) if devs else None),
            }
        # ★ 替代判据:**逐列**与同一扰动的控制臂比(控制臂 = 米制那 6 列的全部逐帧比值)
        for c in screen:
            info = entry["columns"].get(c) or {}
            info["verdict"] = verdict(info.get("_pairs") or [], control_pairs)
        for c in cols:
            entry["columns"][c].pop("_pairs", None)     # 判据算完了,别把逐帧数据塞进 JSON
        result["arms"][arm] = entry
    if (out_dir / f"reframe_{NOISE_ARM}.csv").exists():
        rows = _read(out_dir / f"reframe_{NOISE_ARM}.csv")
        n = min(len(rows), len(ref_rows))
        same = sum(1 for a, b in zip(ref_rows[:n], rows[:n])
                   if all(a.get(k) == b.get(k) for k in cols))
        result["noise_floor"] = {"frames": n, "identical_all_columns": same}
    return result


def median_abs_log(ratios: list[float]) -> float | None:
    """逐帧比值取 |ln| 之后的中位数 = 「这一列在这次扰动下典型变了多少」。"""
    logs = [abs(math.log(r)) for r in ratios if r > 0]
    return statistics.median(logs) if logs else None


def verdict(column: list[float], control: list[float]) -> str:
    """**替代判据**(可机检、不碰 mediapipe)—— 判这一列在**同一扰动**下算不算「稳」。

    输入:
      · `column`  —— 本列逐帧的「扰动后 / 原」比值;
      · `control` —— **同一扰动**下控制臂(米制那 6 个 `*_jitter_world` 列:它们不除肩宽、
        本批的口径对它们是中性的)的逐帧比值 —— 拿它的**全部**列合并进来即可。

    判据:
        中位 |ln(本列比值)|  ≤  **控制臂 |ln 比值| 的 p90**

    为什么是「控制臂的 p90」而不是「≤ 1」(绝对不变):**绝对不变在真像素上不可达** ——
    重取景必然改构图与外侧像素,而模型对这些的响应就是它自己的固有噪声。
    为什么不是「≤ 控制臂的中位数」:6 列之间本身就差 2–3 倍(实测),中位数会把
    "比控制臂略差、但仍在模型自身离散范围内"的列误判成不合格;而 p90 那一条
    仍然能抓住真要防的那一类 —— **跟着取景一起单调变**的列(它的中位偏差会与
    控制臂的 p90 同量级或更大)。

    返回 `"within"` / `"exceeds"` / `"insufficient"`(< 4 对配不出统计量)。
    """
    dev = median_abs_log(column)
    logs = [abs(math.log(r)) for r in control if r > 0]
    if dev is None or len([r for r in column if r > 0]) < 4 or len(logs) < 4:
        return "insufficient"
    bar = quantiles(logs, qs=(0.90,))[0]
    return "within" if dev <= bar else "exceeds"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="重取景归因探针(手势 ② 的两条补做实验)")
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", default="produce", choices=("produce", "analyze"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--noise-arm", action="store_true",
                    help="多跑一条原样腿(噪声地板;只对一场做就够)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser()
    if args.mode == "analyze":
        print(json.dumps(analyze(out_dir), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    from media_retention import recording_dir
    sid = args.session_id
    session_dir = (Path(args.root) / sid) if args.root else recording_dir(sid)
    arms = tuple(a for a in args.arms.split(",") if a)
    if args.noise_arm:
        arms = arms + (NOISE_ARM,)
    n = produce(session_dir, out_dir, arms, sid)
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())

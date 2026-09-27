"""`au6_cheek_raise` 的分母审计:哪个候选尺度真的与 `head_yaw` 无关?

为什么需要这个脚本(L0 行 `au6_cheek_raise` 的 basis 引它):该列的分母换过两次 ——
`face_height`(鼻尖→下巴)→ B2 换成**眼距** `dist(lm[130], lm[359])` → 现在换成
`dist(lm[10], lm[152])`(前额顶→下巴)。B2 那次换完**实测反而更差**(与 `head_yaw`
的会话内相关 −0.017/+0.260/−0.012 → −0.236/+0.508/+0.300)。

"分母选哪个"是**我们定的**(L0 表的 B 档规矩:我们自己定的参数必须写依据),而依据
不能是"试到哪个好就用哪个"。所以这里把候选摆齐、把数字算出来。**两张表缺一不可**:

  · ① 每个候选**自身**与 `head_yaw` 的会话内相关(它随不随头姿动);
  · ② 用每个候选当分母**重算出来的 au6** 与 `head_yaw` 的相关 —— 这才是本列要消掉的病。

另有 ③④ 两条**机制**读数,用来回答"为什么它该稳"(而不只是"它恰好数值更好"):
  · ③ 分子的方向:颊↔眼下缘这一段的 `|Δy|/(|Δx|+|Δy|)`;
  · ④ 端点是不是**突出的鼻尖**:`|Δx|` 与 yaw 的相关。

★ 落盘的是**逐帧原始量**(不是相关系数):相关系数是**读数**,换个判据还要重算,
存进仓库的应该是能重算它的那份素材。

用法:
    PY=~/miniconda3/envs/jingxin/bin/python
    $PY experiments/au6_denominator_audit.py --out-dir /tmp/au6_audit
    # 只出数、不重抽(读上面那份 CSV):
    $PY experiments/au6_denominator_audit.py --out-dir /tmp/au6_audit --report-only

⚠️ 帧与时间戳的取法与 `experiments/replay_retained.py` **同一处**(`load_frames`):
时间戳一律用留存账本里的 `declared_ts`(纪律 3 —— mediapipe 的 VIDEO 模式把时间戳
当模型输入,喂墙钟会得出"推理不可复现"的**假**结论)。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.replay_retained import load_frames          # noqa: E402
from media_retention import recording_dir                    # noqa: E402

# 三场正式素材(M3 的 3 场,见 `docs/资源总览.md`)。
FORMAL_SESSIONS = ("20260926_153202_2b11",
                   "20260926_153854_1592",
                   "20260926_155559_caf0")

# |head_yaw| 超过它的帧按**退化估计**剔除。head_yaw 是 2D 近似(该行自己判「重构」,
# 真 3D 版卡在无相机内参),实测 caf0 有一帧 −8.4(鼻子 x 减双颊中点 x 除以一个很小的
# 面宽)—— 那是坏值,不是"转头 8 倍脸宽"。1.0 这条线远在任何真实转头之外。
YAW_TRIM = 1.0


def _d(lm, i, j) -> float:
    return float(np.linalg.norm(np.asarray(lm[i], dtype=float) - np.asarray(lm[j], dtype=float)))


def _dy(lm, i, j) -> float:
    """**纯竖直**距(只取 y 差)—— yaw 下它一阶不变,euclid 会把横向分量带进来。"""
    return float(abs(np.asarray(lm[i], dtype=float)[1] - np.asarray(lm[j], dtype=float)[1]))


def _cands():
    """候选分母:`(列名, 方向, 取两点的函数)`。方向只用于报告分组(横/竖/近竖直)。"""
    return (
        ("den_eye_dist", "横", _d, 130, 359),
        ("den_face_width", "横", _d, 234, 455),
        ("den_brow_eye", "横", _d, 107, 336),
        ("den_face_height", "竖", _d, 1, 152),        # 旧分母(鼻子端点)
        ("den_forehead_chin", "竖", _d, 10, 152),     # ★ 现用
        ("den_glabella_chin", "竖", _d, 9, 152),
        ("den_browcenter_chin", "竖", _d, 168, 152),
        ("den_eye_vert_l", "竖", _d, 159, 145),
        ("den_eye_vert_r", "竖", _d, 386, 374),
        ("den_eye_vert_mean", "竖(均值)", None, 0, 0),
        ("den_nose_mouth", "竖", _d, 1, 13),
        ("den_iris_span", "横(双眼虹膜中心)", _d, 468, 473),
        ("den_mouth_width", "横(但会随表情变)", _d, 61, 291),
    )


def _cand_value(name: str, fn, i: int, j: int, lm) -> float:
    if name == "den_eye_vert_mean":
        return (_d(lm, 159, 145) + _d(lm, 386, 374)) / 2.0
    return fn(lm, i, j)


def _head_yaw(lm) -> float:
    """`HeadPoseExtractor.extract` 的 head_yaw 分支(逐字:鼻尖 x − 双颊中点 x,÷ face_width)。"""
    nose_tip = np.asarray(lm[1], dtype=float)
    ear_mid = (np.asarray(lm[234], dtype=float) + np.asarray(lm[455], dtype=float)) / 2.0
    return float((nose_tip[0] - ear_mid[0]) / _d(lm, 234, 455))


def extract_session(session_dir: Path, out_csv: Path) -> tuple[int, dict]:
    """对一场会话的留存帧逐帧抽 landmark,把候选分母写成 CSV。返回 (有脸帧数, 缺口报告)。"""
    import cv2
    from face_expression.config import FACE_MODEL
    from face_expression.pipeline.detector import FaceDetector

    frames, report = load_frames(session_dir, "face")
    detector = FaceDetector(FACE_MODEL)
    rows = []
    try:
        for path, ts in frames:
            img = cv2.imread(str(path))
            if img is None:
                print(f"[跳过] 读不出:{path}", file=sys.stderr)
                continue
            lm, _bs = detector.detect_with_blendshapes(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ts)
            if not lm:
                continue                     # 没检出脸的帧不进统计(与各列口径一致)
            row = {
                "file": path.name,
                "ts": ts,
                # `head_yaw` 逐字照 `HeadPoseExtractor.extract` 的口径 ——
                # 要对照的是**本仓那一列**,不是另一个更好看的 yaw。
                "head_yaw": _head_yaw(lm),
                "num_seg_left": _d(lm, 205, 145),
                "num_seg_right": _d(lm, 425, 374),
                # ④ 横向污染探针:同一条竖直跨度的 |Δx|(鼻子端点 vs 轮廓端点)
                "dx_nose_chin": abs(np.asarray(lm[1], float)[0]
                                    - np.asarray(lm[152], float)[0]),
                "dx_forehead_chin": abs(np.asarray(lm[10], float)[0]
                                        - np.asarray(lm[152], float)[0]),
                # ③ 分子的方向:纯竖直差 ÷ (|Δx| + |Δy|),1.0 = 完全竖直
                "num_vertical_share": ((_dy(lm, 205, 145) + _dy(lm, 425, 374))
                                       / (_d(lm, 205, 145) + _d(lm, 425, 374)
                                          + abs(np.asarray(lm[205], float)[0] - np.asarray(lm[145], float)[0])
                                          + abs(np.asarray(lm[425], float)[0] - np.asarray(lm[374], float)[0]))),
            }
            for name, _dirn, fn, i, j in _cands():
                row[name] = _cand_value(name, fn, i, j, lm)
            rows.append(row)
    finally:
        detector.close()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)
    return len(rows), report


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    if x.std() == 0 or y.std() == 0:
        return float("nan")                  # 常量列没有相关可算,如实吐 nan 不吐 0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    from scipy import stats
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def report(csv_paths: dict[str, Path]) -> None:
    per_session = {}
    for tag, p in csv_paths.items():
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        per_session[tag] = {k: np.array([float(r[k]) for r in rows])
                            for k in rows[0] if k != "file"}

    def trim(d):
        m = np.abs(d["head_yaw"]) <= YAW_TRIM
        return {k: v[m] for k, v in d.items()}

    tags = list(per_session)
    cands = _cands()

    print(f"\n=== ③/④ 机制读数(三场;|head_yaw| > {YAW_TRIM} 的退化帧已剔除)===")
    for name, why in (("num_vertical_share",
                       "分子的竖直占比(1.0 = 纯竖直;#本列要归一的是一个竖直量)"),
                      ("dx_nose_chin", "|Δx(鼻尖→下巴)| 与 yaw 的相关(鼻尖突出带来的横向污染)"),
                      ("dx_forehead_chin", "|Δx(前额顶→下巴)| 与 yaw 的相关")):
        if name == "num_vertical_share":
            print(f"  {why}")
            print("      中位数:" + "  ".join(f"{t} {np.median(d[name]):.3f}"
                                              for t, d in per_session.items()))
        else:
            vals = [_pearson(trim(d)["head_yaw"], trim(d)[name]) for d in per_session.values()]
            print(f"  {why}\n      r(yaw): " + "  ".join(f"{t} {v:+.3f}"
                                                         for t, v in zip(tags, vals)))

    print(f"\n=== ① 候选分母**自身**与 head_yaw 的会话内相关(越接近 0 越不随头姿动)===")
    print(f"{'候选':<22}{'方向':<10}" + "".join(f"{t:>11}" for t in tags))
    for name, dirn, _fn, _i, _j in cands:
        vals = [_pearson(d["head_yaw"], d[name]) for d in per_session.values()]
        print(f"{name:<22}{dirn:<10}" + "".join(f"{v:>+11.3f}" for v in vals))
    vals = [_pearson(d["head_yaw"], d["num_seg_left"] + d["num_seg_right"])
            for d in per_session.values()]
    print(f"{'num(分子)':<22}{'竖':<10}" + "".join(f"{v:>+11.3f}" for v in vals))

    print("\n=== ② 以该候选为分母重算的 au6 = 1 − 分子/(2×分母),与 head_yaw 的相关 ===")
    print("    (这一栏才是本列要消掉的病;`Δslope` = 比值对 yaw 的相对灵敏度,")
    print("     即 d ln(分子/分母)/d yaw —— 跨场可比,不受各场 yaw 跨度不同的影响)")
    print(f"{'候选':<22}{'方向':<10}" + "".join(f"{t + ' r|Δ':>18}" for t in tags))
    for name, dirn, _fn, _i, _j in cands:
        cells = []
        for d in per_session.values():
            num = d["num_seg_left"] + d["num_seg_right"]
            a = 1.0 - num / (2.0 * d[name])
            slope = np.polyfit(d["head_yaw"], np.log(num / d[name]), 1)[0]
            cells.append(f"{_pearson(d['head_yaw'], a):+.3f}|{slope:+.3f}")
        print(f"{name:<22}{dirn:<10}" + "".join(f"{c:>18}" for c in cells))

    print("\n=== ②b 稳健性:Spearman(秩相关;不受那几帧退化 yaw 的量级影响)===")
    print(f"{'候选':<22}" + "".join(f"{t:>11}" for t in tags))
    for name, _dirn, _fn, _i, _j in cands:
        vals = []
        for d in per_session.values():
            num = d["num_seg_left"] + d["num_seg_right"]
            vals.append(_spearman(d["head_yaw"], 1.0 - num / (2.0 * d[name])))
        print(f"{name:<22}" + "".join(f"{v:>+11.3f}" for v in vals))

    print("\n=== ③b 各候选的取值形态(判断是否退化:相对离散 std/mean)===")
    for name, _dirn, _fn, _i, _j in cands:
        vals = [f"{d[name].mean():.4f}±{d[name].std() / d[name].mean():.3f}"
                for d in per_session.values()]
        print(f"{name:<22}" + "  ".join(f"{t}: {v}" for t, v in zip(tags, vals)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="au6 分母审计(候选 vs head_yaw)")
    ap.add_argument("--sessions", nargs="*", default=list(FORMAL_SESSIONS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report-only", action="store_true",
                    help="不重抽,只读 --out-dir 里已有的 CSV 出数")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    csv_paths: dict[str, Path] = {}
    for sid in args.sessions:
        tag = sid.rsplit("_", 1)[-1]
        p = out_dir / f"landmarks_{tag}.csv"
        if not args.report_only or not p.exists():
            n, rep = extract_session(recording_dir(sid), p)
            print(f"[{sid}] 盘上 {rep['frames_on_disk']} 帧 / 有时间戳 {rep['frames_with_ts']} 帧 "
                  f"/ 碎行 {rep['torn_lines']} → 抽出有脸帧 {n} → {p}")
        csv_paths[tag] = p
    report(csv_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

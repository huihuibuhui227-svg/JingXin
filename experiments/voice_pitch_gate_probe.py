#!/usr/bin/env python3
"""pitch 那四条(C 档)到底能不能加门控 —— **实测探针**,不是产出方。

为什么需要它:`pitch_mean` / `pitch_variation` / `pitch_p10` / `pitch_p90` 四行的
`definition` 写明了修法 ——「用 `pyin` 的第 3 个返回值 `voiced_prob`(现被 `_` 丢弃)当门控」。
而同一行的 `l0_output` 写着「门控是一个**阈值**,阈值来自标定集 ⟹ 留给 L1」,
`voiced_prob_mean` 那行的 `l0_output` 又留了一句警告:「真素材的值远低于合成正弦段……
**当门控原料用之前先看这条**」。

本脚本就是去**看那一条**:拿三场正式素材的留存音频,把「加门控之后会变成什么」算出来,
好让「该不该落地这个门控」这个判断有数可依 —— 而不是照着 definition 里那句
「修法 = 用 voiced_prob 当门控」直接写进代码。

**它不产出任何 L0 列**,也不改任何代码:
  · 「现状」那一列走**真产出方**(`voice_interaction/api/app.prosody_features_from_pcm`),
    并当场核对它与本脚本自己那份 pyin 的一致(不一致就报,免得两边口径偷偷分叉);
  · 「加门控之后」是**反事实**,只能算 —— 那是 `voiced_prob > 阈值` 这一个动作的全部内容,
    没有别的东西可以"跑真产出方"。

用法:
    PY=~/miniconda3/envs/jingxin/bin/python
    $PY experiments/voice_pitch_gate_probe.py            # 三场正式素材
    $PY experiments/voice_pitch_gate_probe.py --thresholds 0.02 0.05 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa                                                       # noqa: E402

from experiments.replay_retained import load_audio_segments           # noqa: E402

SESSIONS = ("20260926_153202_2b11", "20260926_153854_1592", "20260926_155559_caf0")
ROOT = Path.home() / "shared" / "jingxin_recordings"
# 「噪声帧」的操作定义:pyin 的 fmax 是 C7 = 2093 Hz,人声 f0 到不了 1000 Hz ——
# 表里点名的症状就是 1000 Hz 量级的尖峰,所以用 400 Hz 当分界(与 pitch_mean 的
# acceptance ① 同一条线:那里就是「不再拖到 400 Hz 以上」)。
NOISE_HZ = 400.0


def _extract(path: Path):
    """读一段留存 WAV → 16k 单声道 PCM → (f0, voiced_flag, voiced_prob)。"""
    import wave

    with wave.open(str(path), "rb") as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)
        pcm = wf.readframes(wf.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    f0, flag, prob = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=16000)
    return pcm, f0, flag, prob


def _stats(f0_voiced: np.ndarray) -> dict:
    if len(f0_voiced) == 0:
        return {"n": 0, "mean": None, "p10": None, "p90": None}
    return {"n": len(f0_voiced),
            "mean": round(float(np.mean(f0_voiced)), 2),
            "p10": round(float(np.percentile(f0_voiced, 10)), 2),
            "p90": round(float(np.percentile(f0_voiced, 90)), 2)}


def probe(thresholds: list[float], sessions=SESSIONS, root=ROOT) -> int:
    import importlib

    voice_app = importlib.import_module("voice_interaction.api.app")

    print(f"{'段':34s} {'语音帧':>5} {'现状 mean/p10/p90':>28} | "
          + " | ".join(f"vp>{t:<4} mean(n)" for t in thresholds))
    worst_ungated = (0.0, "")
    worst_gated = {t: (0.0, "") for t in thresholds}
    blanked = {t: 0 for t in thresholds}
    sep_rows: dict[float, list[tuple[float, float]]] = {t: [] for t in thresholds}
    total = 0

    for sid in sessions:
        segs, report = load_audio_segments(root / sid)
        seen: set[str] = set()
        kept = []
        for p, rec in segs:
            if rec.get("source_endpoint") != "/interview/answer_audio":
                continue
            key = str(rec.get("sha256") or p.name)
            if key in seen:
                continue
            seen.add(key)
            kept.append((p, rec))
        for q, (path, _rec) in enumerate(kept):
            pcm, f0, flag, prob = _extract(path)
            fv, pv = f0[flag], prob[flag]
            live = voice_app.prosody_features_from_pcm(pcm)
            mine = _stats(fv)
            # 现状那一格必须与**真产出方**逐字相同 —— 否则下面那些反事实就没法比。
            for k in ("mean", "p10", "p90"):
                if live[f"pitch_{'mean' if k == 'mean' else k}"] != mine[k]:
                    print(f"⚠️ 口径分叉:{path.name} 活路径 "
                          f"{live[f'pitch_{k}']!r} vs 本脚本 {mine[k]!r}", file=sys.stderr)
            tag = f"{sid[-4:]}/Q{q} {path.name}"
            cells = []
            for t in thresholds:
                m = pv > t
                if not m.any():
                    blanked[t] += 1
                    cells.append(f"{'—(空)':>10}(0)")
                    continue
                sub = _stats(fv[m])
                cells.append(f"{sub['mean']:>10}({sub['n']})")
                if sub["mean"] > worst_gated[t][0]:
                    worst_gated[t] = (sub["mean"], tag)
                # 分离度:门控**去掉**的那些帧里,有多大比例是「噪声帧」(f0>400)。
                # 拿它和这一段的噪声帧基准率比 —— 两者相近 ⟹ 门控没有挑对象,
                # 只是在按比例删帧(删掉的多半是好帧)。
                removed = fv[~m]
                if len(removed):
                    prec = float((removed > NOISE_HZ).mean())
                    base = float((fv > NOISE_HZ).mean())
                    sep_rows[t].append((prec, base))
            total += 1
            print(f"{tag:34s} {len(fv):>5} {mine['mean']:>9}/{mine['p10']:>8}/{mine['p90']:>8} | "
                  + " | ".join(cells))
            if mine["mean"] and mine["mean"] > worst_ungated[0]:
                worst_ungated = (mine["mean"], tag)

    print()
    print(f"[现状] 24 段的 `pitch_mean` 最大 = {worst_ungated[0]:.2f} Hz ({worst_ungated[1]})")
    for t in thresholds:
        w = worst_gated[t]
        pairs = sep_rows[t]
        prec = sum(p for p, _ in pairs) / len(pairs) if pairs else float("nan")
        base = sum(b for _, b in pairs) / len(pairs) if pairs else float("nan")
        print(f"[门控 vp>{t}] `pitch_mean` 最大 = {w[0]:8.2f} Hz ({w[1] or '—'})"
              f";整段被删空的段数 = {blanked[t]}/{total}"
              f";去掉的帧里噪声帧占比 = {prec:.3f}(基准率 {base:.3f})")
    print()
    print("判读:①「去掉的帧里噪声帧占比」≈「基准率」⟹ 门控没有挑出噪声,只是在按比例删帧;"
          "\n      ②最大 `pitch_mean` 反而变大 ⟹ 留下的帧比平均数更偏高频;"
          "\n      ③「整段被删空」的那些段是**干净的**(现状 p90 就在 100 Hz 上下)⟹ 门控删错了对象。")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pitch 门控可行性实测(反事实探针)")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--session", action="append", default=None,
                    help="只跑这几场(默认三场正式素材)")
    args = ap.parse_args(argv)
    return probe(args.thresholds, tuple(args.session) if args.session else SESSIONS)


if __name__ == "__main__":
    raise SystemExit(main())

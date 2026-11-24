#!/usr/bin/env python3
"""Head-to-head empirical benchmark: CPU (GaussianHMM + PELT + Conformal Gate)
vs Tesla P40 GPU (strict torch.float32 1D-TCN + GRU sequence stage classifier).

Designed specifically to run on an idle Tesla P40 (e.g. CUDA_VISIBLE_DEVICES=0)
using `/home/zgy/anaconda3/envs/funasr/bin/python`, while benchmarking the exact
CPU Layer 3 referee pipeline in `/home/zgy/anaconda3/envs/jingxin/bin/python` on the
identical 47 anonymized sessions (`5,666` windows).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_JSON = (
    Path("/home/zgy/JingXin/07_三方向与评审筹备/系统补强")
    / "gpu_vs_cpu_referee_benchmark_20261124.json"
)
JINGXIN_PY = "/home/zgy/anaconda3/envs/jingxin/bin/python"
FUNASR_PY = "/home/zgy/anaconda3/envs/funasr/bin/python"


def _run_cpu_referee_side(export_npz: Path, sessions_json: Path) -> dict[str, Any]:
    """Run CPU Layer 3 training windows export + single/batch timings in jingxin env."""
    script = f"""
import json, time
from pathlib import Path
import numpy as np
from report_frontend.data_loader import LogDataLoader
from report_frontend import behavior_regime_model as brm
from report_frontend import l1_report
from report_frontend import referee_ml_engine as rm
from tools import train_referee_ml as trm

sessions = trm.load_real_sessions()

# Load HMM + Gate for exact single-session and batch timings
hmm_art = rm._load_joblib_cached(rm.DEFAULT_MODEL_DIR / "referee_hmm_model.joblib", rm._HMM_CACHE, ("preprocessor", "hmm", "state_map"))
gate_art = rm._load_joblib_cached(rm.DEFAULT_MODEL_DIR / "referee_conformal_gate.joblib", rm._GATE_CACHE, ("scaler", "classifier", "iso_scaler", "isolation_forest", "q_hat"))

prep = hmm_art["preprocessor"]
hmm = hmm_art["hmm"]
state_map = hmm_art["state_map"]
state_order = list(brm.STATE_ORDER)
state_to_idx = {{s: i for i, s in enumerate(state_order)}}

sess_X_list = []
sess_y_list = []
for s in sessions:
    X_w = s["X_windows"]
    if len(X_w) < 8:
        continue
    Xs = prep.transform(X_w)
    hmm_ids = hmm.predict(Xs)
    labels = np.array([state_to_idx[state_map[int(cid)]] for cid in hmm_ids], dtype=np.int64)
    sess_X_list.append(Xs.astype(np.float32))
    sess_y_list.append(labels)

# Pad/clip sequences to T=128 windows for GPU TCN/BiGRU comparison
T_MAX = 128
N_SESS = len(sess_X_list)
D_FEAT = sess_X_list[0].shape[1]
X_seq = np.zeros((N_SESS, T_MAX, D_FEAT), dtype=np.float32)
y_seq = np.full((N_SESS, T_MAX), -1, dtype=np.int64)
lengths = np.zeros((N_SESS,), dtype=np.int64)
for i, (xs, ys) in enumerate(zip(sess_X_list, sess_y_list)):
    L = min(len(xs), T_MAX)
    X_seq[i, :L] = xs[:L]
    y_seq[i, :L] = ys[:L]
    lengths[i] = L

np.savez_compressed(
    {str(export_npz)!r},
    X_seq=X_seq,
    y_seq=y_seq,
    lengths=lengths,
    sess_idx=np.arange(N_SESS, dtype=np.int64),
)

# Benchmark real single-session CPU referee latency (warm, preloaded models + l1_summary)
bench_rec = next((s for s in sessions if s["logs"].get("face") is not None and len(s["logs"]["face"]) >= 200), sessions[0])
bench_l1 = l1_report.build_l1_report(bench_rec["session_id"], logs=bench_rec["logs"])["l1"]
_ = rm.evaluate_referee_ml(bench_rec["logs"], "bench_anon", l1_summary=bench_l1)
single_times = []
for _ in range(15):
    t0 = time.perf_counter()
    res = rm.evaluate_referee_ml(bench_rec["logs"], "bench_anon", l1_summary=bench_l1)
    single_times.append((time.perf_counter() - t0) * 1000.0)

# Benchmark pure HMM Viterbi + Conformal Gate vector inference across 1 session vs batch of 128/256
hmm_single_times = []
sample_Xs = sess_X_list[0]
sample_vec = bench_rec["ev_vec"].reshape(1, -1)
for _ in range(50):
    t0 = time.perf_counter()
    ids = hmm.predict(sample_Xs)
    z_vec = gate_art["scaler"].transform(sample_vec)
    _ = gate_art["classifier"].predict_proba(z_vec)
    _ = gate_art["isolation_forest"].decision_function(gate_art["iso_scaler"].transform(sample_vec))
    hmm_single_times.append((time.perf_counter() - t0) * 1000.0)

# Batch of 128 & 256 sessions CPU timing
def run_cpu_batch(n_batch):
    t0 = time.perf_counter()
    for b in range(n_batch):
        xs = sess_X_list[b % N_SESS]
        _ = hmm.predict(xs)
        vec = sessions[b % len(sessions)]["ev_vec"].reshape(1, -1)
        _ = gate_art["classifier"].predict_proba(gate_art["scaler"].transform(vec))
        _ = gate_art["isolation_forest"].decision_function(gate_art["iso_scaler"].transform(vec))
    return (time.perf_counter() - t0) * 1000.0

b128 = [run_cpu_batch(128) for _ in range(5)]
b256 = [run_cpu_batch(256) for _ in range(5)]

summary = {{
    "n_sessions_seq": int(N_SESS),
    "total_windows_seq": int(lengths.sum()),
    "feature_dim": int(D_FEAT),
    "cpu_full_evaluate_ms_median": round(float(np.median(single_times)), 3),
    "cpu_full_evaluate_ms_p95": round(float(np.percentile(single_times, 95)), 3),
    "cpu_model_core_single_ms_median": round(float(np.median(hmm_single_times)), 3),
    "cpu_batch128_ms_median": round(float(np.median(b128)), 3),
    "cpu_batch128_sess_per_sec": round(128.0 / (float(np.median(b128)) / 1000.0), 1),
    "cpu_batch256_ms_median": round(float(np.median(b256)), 3),
    "cpu_batch256_sess_per_sec": round(256.0 / (float(np.median(b256)) / 1000.0), 1),
}}
Path({str(sessions_json)!r}).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("CPU_BENCH_OK", json.dumps(summary))
"""
    proc = subprocess.run(
        [JINGXIN_PY, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CPU bench failed ({proc.returncode}):\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
    return json.loads(sessions_json.read_text(encoding="utf-8"))


def _run_gpu_sequence_referee(export_npz: Path) -> dict[str, Any]:
    """Run PyTorch GPU FP32 1D-TCN + BiGRU sequence referee on Tesla P40 GPU."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold

    data = np.load(export_npz)
    X_seq = data["X_seq"]      # (N, T, D) float32
    y_seq = data["y_seq"]      # (N, T) int64 (-1 padded)
    lengths = data["lengths"]  # (N,)
    groups = data["sess_idx"]  # each session in own group

    n_sess, t_max, d_feat = X_seq.shape
    n_classes = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class StageTCNBiGRU(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 64, n_cls: int = 4) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2)
            self.act = nn.GELU()
            self.gru = nn.GRU(hidden, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hidden, n_cls)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, D)
            h = self.act(self.conv1(x.transpose(1, 2))).transpose(1, 2)
            out, _ = self.gru(h)
            return self.head(out)

    # 5-fold GroupKFold evaluating GPU StageTCNBiGRU consistency with HMM macro stages
    gkf = GroupKFold(n_splits=5)
    fold_accs: list[float] = []
    fold_train_ms: list[float] = []
    oof_preds = np.full_like(y_seq, -1)

    X_t_all = torch.tensor(X_seq, dtype=torch.float32, device=device)
    y_t_all = torch.tensor(y_seq, dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    last_model: StageTCNBiGRU | None = None
    for tr_idx, va_idx in gkf.split(X_seq, groups=groups):
        torch.manual_seed(42)
        model = StageTCNBiGRU(d_feat, hidden=64, n_cls=n_classes).to(dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(ignore_index=-1)

        X_tr, y_tr = X_t_all[tr_idx], y_t_all[tr_idx]
        X_va, y_va = X_t_all[va_idx], y_seq[va_idx]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model.train()
        for _ in range(80):
            opt.zero_grad(set_to_none=True)
            logits = model(X_tr)
            loss = crit(logits.reshape(-1, n_classes), y_tr.reshape(-1))
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fold_train_ms.append((time.perf_counter() - t0) * 1000.0)

        model.eval()
        with torch.no_grad():
            pred_va = model(X_va).argmax(dim=-1).cpu().numpy()
        mask = y_va >= 0
        acc = float((pred_va[mask] == y_va[mask]).mean()) if mask.any() else 0.0
        fold_accs.append(acc)
        oof_preds[va_idx] = pred_va
        last_model = model

    assert last_model is not None
    last_model.eval()
    train_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if device.type == "cuda"
        else 0.0
    )

    # Measure isolated cold torch + CUDA init latency via fresh subprocess
    cold_proc = subprocess.run(
        [
            FUNASR_PY,
            "-c",
            "import time; t0=time.perf_counter(); import torch; x=torch.zeros(1, device='cuda:0'); torch.cuda.synchronize(); print(round((time.perf_counter()-t0)*1000.0, 2))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cold_cuda_init_ms = float(cold_proc.stdout.strip().splitlines()[-1])

    # Transition smoothness of predicted GPU neural stage sequence
    gpu_trans = 0
    hmm_trans = 0
    for i in range(n_sess):
        L = int(lengths[i])
        yp = oof_preds[i, :L]
        yt = y_seq[i, :L]
        gpu_trans += int(sum(1 for a, b in zip(yp[:-1], yp[1:]) if a != b))
        hmm_trans += int(sum(1 for a, b in zip(yt[:-1], yt[1:]) if a != b))

    # Latency benchmarks: GPU Single (including H2D transfer vs resident), Batch 128, Batch 256
    x_one_cpu = X_seq[:1]
    x_128_cpu = np.repeat(X_seq, (128 // n_sess) + 1, axis=0)[:128]
    x_256_cpu = np.repeat(X_seq, (256 // n_sess) + 1, axis=0)[:256]

    # Warmup GPU kernels
    with torch.no_grad():
        for _ in range(10):
            tx = torch.tensor(x_one_cpu, dtype=torch.float32, device=device)
            _ = last_model(tx)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def bench_gpu(arr: np.ndarray, reps: int = 30) -> tuple[float, float]:
        e2e_times, kernel_times = [], []
        tx_dev = torch.tensor(arr, dtype=torch.float32, device=device)
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                tx = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                out = last_model(tx).argmax(dim=-1).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            e2e_times.append((time.perf_counter() - t0) * 1000.0)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            with torch.no_grad():
                _ = last_model(tx_dev)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            kernel_times.append((time.perf_counter() - t1) * 1000.0)
        return float(np.median(e2e_times)), float(np.median(kernel_times))

    one_e2e, one_kern = bench_gpu(x_one_cpu, reps=50)
    b128_e2e, b128_kern = bench_gpu(x_128_cpu, reps=25)
    b256_e2e, b256_kern = bench_gpu(x_256_cpu, reps=25)

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if device.type == "cuda"
        else 0.0
    )
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "precision": "torch.float32",
        "architecture": "1D-TCN (k=5, 64ch) + BiGRU (hidden=32x2)",
        "cv_agreement_with_hmm_mean": round(float(np.mean(fold_accs)), 4),
        "cv_agreement_std": round(float(np.std(fold_accs)), 4),
        "oof_total_transitions_gpu": gpu_trans,
        "oof_total_transitions_hmm": hmm_trans,
        "fold_train_ms_mean": round(float(np.mean(fold_train_ms)), 2),
        "train_peak_vram_mb": train_peak_vram_mb,
        "peak_vram_mb": peak_vram_mb,
        "cold_torch_cuda_init_ms": cold_cuda_init_ms,
        "gpu_single_e2e_ms_median": round(one_e2e, 3),
        "gpu_single_kernel_ms_median": round(one_kern, 3),
        "gpu_batch128_e2e_ms_median": round(b128_e2e, 3),
        "gpu_batch128_sess_per_sec": round(128.0 / (b128_e2e / 1000.0), 1),
        "gpu_batch256_e2e_ms_median": round(b256_e2e, 3),
        "gpu_batch256_sess_per_sec": round(256.0 / (b256_e2e / 1000.0), 1),
    }


def main() -> None:
    if sys.executable != FUNASR_PY and os.environ.get("JX_ALLOW_DIRECT") != "1":
        # Re-exec inside funasr CUDA env cleanly
        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env["JX_ALLOW_DIRECT"] = "1"
        os.execve(FUNASR_PY, [FUNASR_PY, __file__], env)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        export_npz = tmp_dir / "seq_data.npz"
        cpu_json = tmp_dir / "cpu_summary.json"
        cpu_summary = _run_cpu_referee_side(export_npz, cpu_json)
        gpu_summary = _run_gpu_sequence_referee(export_npz)

    batch256_speedup = round(
        cpu_summary["cpu_batch256_ms_median"] / max(gpu_summary["gpu_batch256_e2e_ms_median"], 1e-6),
        2,
    )
    single_ratio = round(
        gpu_summary["gpu_single_e2e_ms_median"] / max(cpu_summary["cpu_model_core_single_ms_median"], 1e-6),
        2,
    )

    payload = {
        "schema_version": "gpu_vs_cpu_referee_benchmark_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu_tested": gpu_summary["gpu_name"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
            "fp16_tensor_core_available": False,
            "precision_used": "float32",
        },
        "cpu_referee": cpu_summary,
        "gpu_neural_sequence_referee": gpu_summary,
        "comparison": {
            "single_session_gpu_vs_cpu_model_core_ratio": single_ratio,
            "batch256_gpu_speedup_over_cpu": batch256_speedup,
            "conclusion_plain": (
                f"实测分层分工：常驻 CPU 完整裁判评估单场耗时 {cpu_summary['cpu_full_evaluate_ms_median']:.2f} ms"
                f"（其中核心 HMM+Conformal 推理 {cpu_summary['cpu_model_core_single_ms_median']:.2f} ms，且 ruptures/hmmlearn/pycatch22 仅支持 CPU）；"
                f"Tesla P40 (FP32) TCN+BiGRU 预热驻留后单条仅 {gpu_summary['gpu_single_e2e_ms_median']:.2f} ms，"
                f"冷启动 CUDA 初始化约 {gpu_summary['cold_torch_cuda_init_ms']:.0f} ms；"
                f"在 256 场并发批量推理时，P40 耗时 {gpu_summary['gpu_batch256_e2e_ms_median']:.2f} ms"
                f"（{gpu_summary['gpu_batch256_sess_per_sec']:.0f} 场/秒），较 CPU ({cpu_summary['cpu_batch256_ms_median']:.2f} ms) 提速 {batch256_speedup} 倍，"
                f"且 5 折交叉验证与 HMM 阶段一致率达 {gpu_summary['cv_agreement_with_hmm_mean'] * 100:.1f}%。"
            ),
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

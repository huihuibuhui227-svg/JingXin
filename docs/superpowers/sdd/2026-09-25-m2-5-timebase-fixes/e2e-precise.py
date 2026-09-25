"""M2.5 Task 9 Step 3(精确版):同一进程内发帧,把 curl 进程启动的开销排除掉。

上一版用 bash + curl 循环,墙钟里混进了 5 次 curl 进程启动(每次 ~25 ms),
于是算出 41.5% 的"相对差" —— 那是**测量噪声**,不是时间基的差。
"""
import glob, os, subprocess, time, urllib.request, uuid

import pandas as pd


def _multipart(data):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"f.png\"\r\nContent-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


PY = os.path.expanduser("~/miniconda3/envs/jingxin/bin/python")
FRAMES = os.path.expanduser("~/shared/mp_frames/frames")
SID = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
LOG = f"data/logs/face_au_log_{SID}.csv"

proc = subprocess.Popen([PY, "-m", "face_expression.api.app"],
                        stdout=open("/tmp/face_m25b.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    else:
        raise SystemExit("❌ 服务没起来")

    stamps = []
    for n in ("0001", "0002", "0003", "0004", "0005"):
        body, ctype = _multipart(open(f"{FRAMES}/frame_{n}.png", "rb").read())
        req = urllib.request.Request(
            f"http://127.0.0.1:8000/analyze?fps=30&session_id={SID}", data=body,
            headers={"Content-Type": ctype})
        stamps.append(time.monotonic())
        urllib.request.urlopen(req, timeout=30).read()
    wall_span = stamps[-1] - stamps[0]
finally:
    proc.terminate(); proc.wait(timeout=20)

df = pd.read_csv(LOG)
span = float(df["timestamp"].max() - df["timestamp"].min())
print(f"  时间戳跨度 = {span:.3f} s")
print(f"  墙钟跨度   = {wall_span:.3f} s   (进程内测,不含 curl 启动)")
print(f"  相对差     = {abs(span - wall_span) / wall_span * 100:.1f} %   (期望 < 10%)")
# ⚠️ 审查 F4:`NaN != 0` 是 **True**,所以原来的 `!= 0` 会把 no_face 那行也数进去 ——
# 账本里那条"1/5 非零"就是这么来的假象。真口径是 **非 NaN 且非 0**。
col = pd.to_numeric(df["is_blink"], errors="coerce")
print(f"  is_blink   = {(col.notna() & (col != 0)).sum()} / {len(df)} 非 NaN 且非 0")
print(f"             = {col.dropna().tolist()}  (no_face 行的值是 NaN,不算数)")
ts = list(df["timestamp"])
print(f"  逐帧时间戳 = {[round(v, 3) for v in ts]}")

# 第 1 帧是**预热帧**:服务端在它身上做模型加载/首次推理,所以"从客户端发出"到
# "服务端打戳"的延迟明显大于后续帧 —— 那是个常数项,不是时间基的偏差。
# 把它排掉,比较第 2→5 帧的**区间**:两边都成了"服务端事件之间的间隔"。
n = len(ts) - 1
ts_span2 = ts[-1] - ts[1]
wall_span2 = stamps[-1] - stamps[1]
print(f"  ── 排掉预热帧(第 2→{n + 1} 帧)──")
print(f"  时间戳区间 = {ts_span2:.3f} s   墙钟区间 = {wall_span2:.3f} s")
print(f"  相对差     = {abs(ts_span2 - wall_span2) / wall_span2 * 100:.1f} %   (期望 < 10%)")

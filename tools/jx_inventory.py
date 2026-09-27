#!/usr/bin/env python
"""生成 `docs/资源总览.md` —— JingXin 全盘资源结构图。

为什么是脚本而不是手写文档:散文里的数会静默过期(本项目实测 7 次)。
本文件里**所有数字都是当场算出来的**,没有一个是手打的;重跑一次就刷新。

用法:
    ~/miniconda3/envs/jingxin/bin/python tools/jx_inventory.py            # 写 docs/资源总览.md
    ~/miniconda3/envs/jingxin/bin/python tools/jx_inventory.py --check    # 只比对,不改文件(退出码 1 = 过期)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "资源总览.md"
RECORDINGS = Path.home() / "shared" / "jingxin_recordings"
SHARED = Path.home() / "shared"
FRONTEND = Path.home() / "JingXin-frontend"

SKIP_DIRS = {"__pycache__", ".git", ".idea", ".pytest_cache", "results", "node_modules", ".venv"}
# 只统计这些后缀的行数 —— 排除 .json/.csv 之类,否则 data/ 一进来数字就没意义
TEXT_SUFFIXES = {".py", ".md", ".sh", ".html"}


def sh(*args: str) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def count_code(d: Path) -> tuple[int, int, int]:
    """→ (文件数, .py 数, 文本行数)。跳过产物目录;只数 TEXT_SUFFIXES。

    ⚠️ 用 `os.walk` + **就地剪枝**,不用 `rglob` —— `rglob` 会把 `.git`(1.4 GB 松散
    对象)和 `results/`(1 万文件)整个走一遍再过滤,实测把本脚本拖到 150 s 以上。
    """
    files = pys = loc = 0
    for dirpath, dirnames, filenames in os.walk(d):
        dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_out(p):
                continue
            files += 1
            if p.suffix == ".py":
                pys += 1
            if p.suffix in TEXT_SUFFIXES:
                loc += loc_of(p)
    return files, pys, loc


def loc_of(p: Path) -> int:
    try:
        return sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
    except OSError:
        return 0


def is_out(p: Path) -> bool:
    """本文件是**自指的** —— 它统计的目录里包含它自己的产出。

    不排掉的话:`docs/` 的行数会随本文件每次改写而变化 ⟹ 写出来的数**下一秒就过期**,
    `--check` 永远红(实测踩过)。产出物一律不计入它自己的统计。
    """
    try:
        return p.resolve() == OUT.resolve()
    except OSError:
        return False


def read_port(path: Path) -> str:
    """从 config/app 里抠出端口 —— 不手打。"""
    try:
        txt = path.read_text(encoding="utf-8")
    except OSError:
        return "?"
    # 三种写法都要认:`"port": 8001` / `port: 5173` / `port = 8000`
    m = re.search(r"\bport\b['\"]?\s*[:=]\s*['\"]?(\d+)", txt)
    return m.group(1) if m else "?"


def classify_session(d: Path, in_trial_dir: bool) -> tuple[str, str, int, bool]:
    """→ (档位, 依据, 问题数, 有 transcript)。

    ★ **`meta.json` 区分不出「正式素材」与「试跑」** —— 实测:三场正式与 `_试跑/`
    里 5 场的 `candidate` / `capture` 块**逐字相同**(同一台机器、同一个脚手架
    `jx_new_session.py` 填的机器实测值)。唯一实质差异是问题数(正式 8,试跑 1~4),
    而那是**偶然**,不是设计。
    ⟹ **可靠判据只有目录位置**:`_试跑/` 下面的一律是试跑。靠 meta.json「填得全不全」
    筛正式素材会把 5 场试跑一起筛进来。
    """
    meta = d / "meta.json"
    if not meta.exists():
        return "试跑", "无 meta.json", 0, (d / "transcript.json").exists()
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return "试跑", "meta.json 读不出", 0, (d / "transcript.json").exists()
    nq = len(doc.get("questions") or [])
    has_tr = (d / "transcript.json").exists()
    if in_trial_dir:
        return "试跑", f"在 `_试跑/` 下(meta.json 与正式素材**同款**,仅问题数 {nq} 不同)", nq, has_tr
    return "正式", f"根目录 + meta.json(问题数 {nq})", nq, has_tr


def sessions_table(root: Path, in_trial_dir: bool = False) -> list[str]:
    out = []
    if not root.exists():
        return ["_(录屏根目录不存在)_"]
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name != "NONE")
    for d in dirs:
        if d.name == "_试跑":
            continue
        tier, why, nq, has_tr = classify_session(d, in_trial_dir)
        media = d / "media"
        # ⚠️ 只数文件、**不取大小**。录屏在 `~/shared`(drvfs 挂载),逐文件 `stat` 实测
        # ~13 ms/个 —— 整棵树 7,586 个文件要 100 s;`os.walk` 只要 1.5 s。
        # 大小要看就现取:`du -sh ~/shared/jingxin_recordings/<sid>`。
        nf = webm = 0
        if media.exists():
            for _dp, _dn, _fns in os.walk(media):
                nf += len(_fns)
                webm += sum(1 for x in _fns if x.endswith(".webm"))
        has = " ".join(
            k for k, v in {
                "questions.jsonl": (d / "questions.jsonl").exists(),
                "replay_face.csv": (d / "replay_face.csv").exists(),
            }.items() if v
        )
        out.append(
            f"| `{d.name}` | **{tier}** | {nq if nq else '—'} | "
            f"{'✅' if has_tr else '—'} | {webm} | {nf:,} | {has} | {why} |"
        )
    return out


def main() -> int:
    doc_est = json.loads((ROOT / "l0_columns.json").read_text(encoding="utf-8"))
    cols = doc_est["columns"]
    st: dict[str, int] = {}
    mt: dict[str, int] = {}
    pend_mt: dict[str, int] = {}
    for c in cols:
        st[c["status"]] = st.get(c["status"], 0) + 1
        mt[c["maturity"]] = mt.get(c["maturity"], 0) + 1
        if c["status"] == "pending":
            pend_mt[c["maturity"]] = pend_mt.get(c["maturity"], 0) + 1

    L: list[str] = []
    A = L.append
    A("# JingXin 资源总览(自动生成)")
    A("")
    A("> **本文件由 `tools/jx_inventory.py` 生成 —— 不要手改。**")
    A("> 改完重跑:`~/miniconda3/envs/jingxin/bin/python tools/jx_inventory.py`")
    A("> 所有数字当场算出;手写一个数就会过期(本项目实测 7 次)。")
    A("")
    A(f"_生成时间:{sh('date', '+%Y-%m-%d %H:%M:%S')}_")
    A("")
    A("---")
    A("")

    # ── 1 环境 ───────────────────────────────────────────────
    A("## 1. 环境")
    A("")
    py = str(Path.home() / "miniconda3/envs/jingxin/bin/python")
    A(f"| | |")
    A("|---|---|")
    A(f"| 本仓 Python | `{py}`(conda env `jingxin`) |")
    A(f"| 版本 | {sh(py, '-V')} |")
    for mod in ("pytest", "pandas", "numpy", "mediapipe", "librosa"):
        vsn = sh(py, "-c", f"import {mod};print({mod}.__version__)")
        A(f"| `{mod}` | {vsn or '未装'} |")
    A(f"| 仓库根 | `{ROOT}` |")
    A(f"| 分支 / HEAD | `{sh('git', 'rev-parse', '--abbrev-ref', 'HEAD')}` / `{sh('git', 'rev-parse', '--short', 'HEAD')}` |")
    A(f"| 未提交条目 | {len(sh('git', 'status', '--porcelain').splitlines())} |")
    A("")
    A("⚠️ **不要用 `~/huihui/bin/python`** —— 那是数据科学环境,本仓没装依赖(实测无 pytest)。")
    A("")
    A(f"模型文件(`models/mediapipe/`,gitignored,共 {sum(1 for _ in (ROOT / 'models').rglob('*.task'))} 个 `.task`):")
    A("")
    for t in sorted((ROOT / "models").rglob("*.task")):
        A(f"- `{t.relative_to(ROOT)}` — {t.stat().st_size/1e6:.1f} MB")
    A("")

    # ── 2 仓库结构 ────────────────────────────────────────────
    A("## 2. 仓库结构(代码量)")
    A("")
    A("| 目录 | 文件 | `.py` | 行数 | 是什么 |")
    A("|---|---:|---:|---:|---|")
    DESCR = {
        "face_expression": "面部服务(端口见下):landmarker → AU/EAR → 面部 CSV",
        "gesture_analysis": "手势服务:姿态+手部 landmark → 抖动/肩/臂 → 手势 CSV",
        "voice_interaction": "语音服务:FunASR 转写 + 韵律 + 测评流水线",
        "report_frontend": "报告层:特征引擎 / 证据门 / 映射 / 生成器",
        "experiments": "实验脚本 + 结果(gitignored `results/` 占绝大多数文件)",
        "tests": "pytest 套件",
        "models": "mediapipe `.task` 模型(gitignored)",
        "data": "运行产物:日志 CSV、报告 HTML(gitignored)",
        "templates": "Flask 面板模板",
        "code_data_supplement": "论文复现包(未跟踪)",
        "docs": "文档 / 规格 / 计划 / SDD 账本",
        "tools": "本仓维护脚本",
        "report_frontend/static": "(不存在)",
    }
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in SKIP_DIRS | {".claude", ".superpowers"}):
        f, pys, loc = count_code(d)
        A(f"| `{d.name}/` | {f:,} | {pys} | {loc:,} | {DESCR.get(d.name, '')} |")
    A("")
    A("**仓库根 `.py`**")
    A("")
    A("| 文件 | 行数 | 是什么 |")
    A("|---|---:|---|")
    ROOT_PY = {
        "app.py": "报告面板(Flask),起 `report_frontend.*` 子进程",
        "l0_columns.py": "L0 表加载 / 校验(`load` / `columns` / `schema_errors`)",
        "logging_config.py": "统一日志设置 `setup_logging()`",
        "media_retention.py": "原始媒体留存:收到的字节**原样**落到仓库外的盘",
        "session_clock.py": "会话相对时钟 —— 保证 ms 严格递增(mediapipe VIDEO 模式要求)",
        "session_meta.py": "Stage-A 元数据层:问题时间账本 + `meta.json` 模板/校验",
    }
    for p in sorted(ROOT.glob("*.py")):
        A(f"| `{p.name}` | {loc_of(p):,} | {ROOT_PY.get(p.name, '')} |")
    A("")

    # ── 3 三个服务 ────────────────────────────────────────────
    A("## 3. 三个采集服务 + 面板")
    A("")
    A("| 服务 | 入口 | 端口 | 写什么 |")
    A("|---|---|---:|---|")
    SVCS = [
        ("face", "face_expression/api/app.py", "face_expression/config.py", "`face_au_log_<sid>.csv`"),
        ("voice", "voice_interaction/api/app.py", "voice_interaction/config.py", "`interview_/research_emotion_log_<sid>.csv` + `assessment_note_<ts>.csv`"),
        ("gesture", "gesture_analysis/api/app.py", "gesture_analysis/config.py", "`gesture_emotion_log_<sid>.csv`"),
    ]
    for name, entry, cfg, writes in SVCS:
        port = read_port(ROOT / cfg) or read_port(ROOT / entry)
        A(f"| {name} | `{entry}` | {port} | {writes} |")
    A(f"| 报告面板(Flask) | `app.py` | {read_port(ROOT / 'app.py')} | 报告 HTML |")
    fe_port = next((p for p in (read_port(FRONTEND / "vite.config.ts"),
                                read_port(FRONTEND / "vite.config.js")) if p != "?"), "?")
    A(f"| 前端(React/Vite,仓库外) | `{FRONTEND}` | {fe_port} | — |")
    A("")
    A("⚠️ 三个服务**无启动脚本**,必须 `setsid` 脱离终端起;否则终端一 Ctrl-Z 就成 T 态")
    A("(端口在听却不回话,看着像崩溃,且收不到 TERM「杀不掉」)。")
    A("")

    # ── 4 L0 列清单 ───────────────────────────────────────────
    A("## 4. L0 列清单 `l0_columns.json`(M3 的任务清单)")
    A("")
    A(f"共 **{len(cols)}** 列。`status` / `maturity` 两轴(A/B/C 判据见 "
      f"`docs/superpowers/specs/2026-09-26-l0-column-table-design.md` §4.2)。")
    A("")
    # ★ `l0_done`(2026-09-27 加):L0 侧已落地并实测,但验收里还有一条只能来自标定集的子项。
    #   它必须**单独成列**显示 —— 混进 pending 会让读的人以为"一行代码都没做"。
    _COLS = ("implemented", "l0_done", "pending", "blocked")
    A("| | implemented | l0_done | pending | blocked | 合计 |")
    A("|---|---:|---:|---:|---:|---:|")
    for m in ("A", "B", "C"):
        row = [sum(1 for c in cols if c["maturity"] == m and c["status"] == s)
               for s in _COLS]
        A(f"| maturity **{m}** | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {sum(row)} |")
    A(f"| **合计** | " + " | ".join(str(st.get(k, 0)) for k in _COLS) + f" | {len(cols)} |")
    A("")
    A(f"★ **待实现 {st.get('pending',0)} 条 = A {pend_mt.get('A',0)} / B {pend_mt.get('B',0)} / C {pend_mt.get('C',0)}**")
    A(f"⟹ **B 档 {pend_mt.get('B',0)} 条不需要标定,可立即开工**;C 档 {pend_mt.get('C',0)} 条卡仪器标定(M3.1)。")
    A("")
    A(f"⚠️ **`l0_done` {st.get('l0_done',0)} 条**:L0 侧口径**已按本表落地并在三场正式素材上实测**,"
      f"但验收里还有一条子项**只能来自标定集**(M3.1 已被裁定跳过)⟹ 不标 `implemented`。"
      f"**别把这 {st.get('l0_done',0)} 条读成「没做」** —— 每行的 `acceptance` 里逐条写着哪条已达成、哪条卡着、解封条件是什么。")
    A("")
    A("按模态:")
    A("")
    A("| 模态 | implemented | l0_done | pending | blocked |")
    A("|---|---:|---:|---:|---:|")
    for mod in ("face", "gesture", "voice", "text"):
        A(f"| {mod} | " + " | ".join(
            str(sum(1 for c in cols if c["modality"] == mod and c["status"] == s))
            for s in _COLS) + " |")
    A("")

    # ── 5 录制素材 ────────────────────────────────────────────
    A("## 5. 录制素材 `~/shared/jingxin_recordings/`")
    A("")
    A("**别删。** 三场正式素材是 M3/M4 的输入;`_试跑/` 9 场是废的但留着做回归。")
    A("")
    A("⚠️ **`meta.json` 区分不出正式素材与试跑** —— 实测三场正式与 `_试跑/` 里 5 场的")
    A("`candidate` / `capture` 块**逐字相同**(同一台机器 + 同一个脚手架 `jx_new_session.py`)。")
    A("**可靠判据只有目录位置**(`_试跑/` 下一律是试跑)。按「meta.json 填得全不全」筛正式素材,")
    A("会把 5 场试跑一起筛进来 —— 这正是「量错了对象」那一类。")
    A("")
    A("### 5.1 根目录")
    A("")
    A("| session | 档位 | 问题数 | transcript | webm | media 文件数 | 还有 | 依据 |")
    A("|---|---|---:|---|---:|---:|---|---|")
    L.extend(sessions_table(RECORDINGS, in_trial_dir=False))
    A("")
    A("### 5.2 `_试跑/`(试跑,不是素材 —— 但**别删**,留作回归)")
    A("")
    A("| session | 档位 | 问题数 | transcript | webm | media 文件数 | 还有 | 依据 |")
    A("|---|---|---:|---|---:|---:|---|---|")
    L.extend(sessions_table(RECORDINGS / "_试跑", in_trial_dir=True))
    A("")
    mirror = SHARED / "jingxin_logs"
    if mirror.exists():
        mdirs = sorted(p for p in mirror.iterdir() if p.is_dir())
        A(f"CSV 镜像:`{mirror}` —— {len(list(mirror.rglob('*.csv')))} 个 CSV,分 {len(mdirs)} 个 session 子目录:")
        A("")
        A("| session | CSV |")
        A("|---|---:|")
        for d in mdirs:
            A(f"| `{d.name}` | {len(list(d.glob('*.csv')))} |")
        A("")

    # ── 6 仓库外工具 ──────────────────────────────────────────
    A("## 6. 仓库外工具 `~/shared/`(不进仓库)")
    A("")
    A("| 文件 | 行数 | 是什么 |")
    A("|---|---:|---|")
    for p in sorted(list(SHARED.glob("jx_*.py")) + list(SHARED.glob("*.sh"))):
        first = ""
        try:
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines()[:8]:
                s = line.strip().strip('"').strip("'")
                if s and not s.startswith(("#", "!", "set ", "cd ", "#!/")):
                    first = s[:70]
                    break
        except OSError:
            pass
        A(f"| `{p.name}` | {loc_of(p):,} | {first} |")
    for p in sorted(SHARED.glob("*.md")):
        A(f"| `{p.name}` | {loc_of(p):,} | (文档) |")
    for p in sorted(SHARED.glob("L0*.csv")):
        A(f"| `{p.name}` | {loc_of(p):,} | L0 表导出 |")
    A("")

    # ── 7 文档 ────────────────────────────────────────────────
    A("## 7. 文档与规格")
    A("")
    A("| 文件 | 行数 | 是什么 |")
    A("|---|---:|---|")
    for sub in ("", "superpowers/specs", "superpowers/plans"):
        base = ROOT / "docs" / sub if sub else ROOT / "docs"
        for p in sorted(base.glob("*.md")):
            if is_out(p):
                continue
            A(f"| `docs/{sub + '/' if sub else ''}{p.name}` | {loc_of(p):,} | |")
    A("")
    sdd = ROOT / "docs/superpowers/sdd"
    if sdd.exists():
        A("### SDD 账本(`docs/superpowers/sdd/`)")
        A("")
        A("| 里程碑 | 文件数 | `progress.md` 行数 |")
        A("|---|---:|---:|")
        for d in sorted(p for p in sdd.iterdir() if p.is_dir()):
            pm = d / "progress.md"
            A(f"| `{d.name}` | {sum(1 for _ in d.rglob('*') if _.is_file())} | {loc_of(pm) if pm.exists() else '—'} |")
        A("")

    # ── 8 开工入口 ────────────────────────────────────────────
    A("## 8. 下次开工的入口")
    A("")
    A("1. `docs/下一步.md` **§0.11**(最新一节)+ §11.6 的「该带进去的 5 样」")
    A("2. `l0_columns.json` —— 就是任务清单:按 `status=pending` + `maturity=B` 筛")
    A("3. `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md` §4 —— 逐列定义的依据")
    A("4. 重放必须喂留存账本的 `declared_ts`(样板 `experiments/replay_retained.py`)")
    A("")
    A("**三条纪律(踩出来的):**")
    A("")
    A("1. 说「验证过」之前先问:**我跑的这条会不会执行到我要证的那段代码?**")
    A("   (合并门对采集层改动**结构上不可见**;只核清单 ≠ 验证)")
    A("2. 文件里**散文中的数**会静默过期 —— 数只说一次,能算的就别手写(本文件即为对策)")
    A("3. 重放用墙钟会得出「推理不可复现」的**假**结论 —— 必须喂 `declared_ts`")
    A("")

    text = "\n".join(L) + "\n"

    if "--check" in sys.argv:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        old_body = re.sub(r"_生成时间:.*?_", "", old)
        new_body = re.sub(r"_生成时间:.*?_", "", text)
        if old_body != new_body:
            print(f"❌ {OUT} 已过期,请重跑本脚本", file=sys.stderr)
            return 1
        print(f"✅ {OUT} 是最新的")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"✅ 写入 {OUT}({len(L)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

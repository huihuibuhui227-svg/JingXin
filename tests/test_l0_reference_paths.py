# tests/test_l0_reference_paths.py
"""`l0_columns.json` 里**引用**的钉子(2026-09-26 M3.0 整支复核 Important 4)。

## 为什么需要它

表里逐行写着「实现/口径在哪个文件的哪几行」—— 那是**依据**,不是装饰。而这批引用的
历史是一部**漂移史**:Task 5 一个文件改了 4 次,引用就漂了 **47 → 30 → 35 → 29**;
最毒的一次是**「按 basename 匹配」把三个本任务没碰过的文件的行号一起平移了 +7**
(Task 5 复核 Important 1)。复核者数出:改正之前,全表 508 处 `文件:行号` 引用里
**140 处(27.6%)名字在仓里不唯一** —— 裸 `app.py` 命中 **5 个**仓内文件、`config.py` 3 个、
`logger.py` 3 个、`api/app.py` 3 个。**「名字不唯一」正是那次误改的机制。**

## 这颗钉子钉什么(以及**不**钉什么)

1. **每个文件引用必须解析到唯一文件**:引用名(见下「什么算一处引用」)在本仓里
   必须**恰好**命中一个文件;命中 0 个或多个 ⟹ 红,除非它在下面的 `EXEMPT` 里
   逐条写明了理由。⟹ 「名字不唯一」这个机制**从此不可能再静默存在**。
2. **带行号的引用必须在界内**:`行号` 或 `起-止` 不超过那个文件的行数。⟹ 文件变短、
   引用指到文件尾之外 ⟹ 红。
3. **豁免表不许变成垃圾桶**:`EXEMPT` 的每个键都必须在表里**真的出现**
   (闲置的豁免 ⟹ 红)。
4. **紧随 `文件:行号` 之后的裸 `:行号`**(形如 `voice_interaction/api/app.py:605 / :867`)
   —— 这是**无歧义**的一种裸号:它的目标就是紧挨在它前面的那个文件。这类也一并核界内。

## ⚠️ 边界(如实说,免得读的人以为它管得比实际宽)

- **什么算一处引用**:一个**带已知扩展名**的路径(`.py/.md/.json/.jsonl/.csv/...`),
  后面可跟 `:行号` 或 `:起-止`。大小写敏感、只认这些扩展名。
- **正文里的「冒号 + 数字」不认**:例如 `依据栏:7 个视线列`、`§4.1:202`、`§5.4:354`。
  这些**不是**文件引用(前者是句子里的冒号,后两者是**文档章节号**)—— 它们与真正的
  裸引用在文本上无法区分,归 N2 那条「校验器覆盖裸号」,当时就被搁置、本轮裁定里也没要求。
  ★ 为了不把这件事藏起来:**实测**这类形体在表里有 ~91 处(去重 42 种),绝大多数是
  `§x.y:NN` 章节号;上面第 4 条只覆盖**紧随文件之后**、因而无歧义的那 4 处。
- **豁免只在「解析不到」这一侧生效**:`EXEMPT` 里的名字如果**后来被提交进仓**
  (例如那两份 `docs/superpowers/**` 文档),豁免自动失效、引用转为正常校验 ——
  豁免**不会**因为文件被提交而误红,但也**因此**抓不到「该删的豁免没删」这一种。
  只有「豁免名字在表里根本不存在」会红(第 3 条)。
- **行号搬了但内容没搬**抓不到:本钉子只判「解析到唯一文件 + 界内」,不判「那一行还是
  那句话」。判内容只能按内容重定(那一步是提交前的人工动作,基准是**任务开始前的 commit**,
  不是 HEAD —— 拿 HEAD 当基准等于跟自己对)。⟹ **本钉子保证的是「不会指到不存在的文件/
  文件尾之外」,不是「指的那一行还对」。** 两次实测的漂移都是后者,所以它**不替代**人工核。

## 红法(都是生产改动)

- 往表里写一个**名字在仓里不唯一**的裸引用(例如把某处的 `gesture_analysis/utils/logger.py`
  改成裸 `logger.py`)⟹ 第 1 条红;
- 把某处的行号改成超出那个文件行数的数 ⟹ 第 2 条红;
- 往 `EXEMPT` 加一个表里根本不出现的名字 ⟹ 第 3 条红;
- 把 `voice_interaction/api/app.py:605 / :867` 里的 `:867` 改成 `:9999` ⟹ 第 4 条红。
"""
import collections
import importlib
import re
import subprocess
from pathlib import Path

l0 = importlib.import_module("l0_columns")

REPO = Path(l0.TABLE_PATH).resolve().parent

# 只认这些扩展名 —— 这条限定同时把 `§4.1:202` / `依据栏:7` 这类正文冒号挡在外面
# (它们的「扩展名」是 `1`/`7` 这种数字,不在名单里)。⚠️ `jsonl` 必须排在 `json` 前面,
# 且后面要跟 `(?![A-Za-z0-9_])` —— 否则 `questions.jsonl` 会被切成 `questions.json`
# (2026-09-26 实测:第一版就这么错过)。
_EXT = "py|pyi|jsonl|json|csv|tsv|md|txt|html|js|jsx|ts|tsx|yaml|yml|toml|sh|cfg|ini|vue|css|sql"
_REF = re.compile(r"([A-Za-z0-9_\-./]*[A-Za-z0-9_\-]\.(?:%s))(?![A-Za-z0-9_])"
                  r"(?::(\d+)(?:-(\d+))?)?" % _EXT)
# `file:NN / :MM` 或 `file:NN,:MM` —— 紧随其后的裸号,目标就是**前面那个文件**
# (整条一起匹配,这样裸号归属谁不靠猜:前一个文件就是捕获组 1)
_NAKED = re.compile(r"([A-Za-z0-9_\-./]*[A-Za-z0-9_\-]\.(?:%s))(?![A-Za-z0-9_])"
                    r":(\d+)(?:-(\d+))?\s*[/,]\s*:(\d+(?:-\d+)?)" % _EXT)

# ── 豁免表:解析不到仓内唯一文件、但**有正当理由**的引用,逐条写理由 ────────────────
# 两类:① 仓库外的运行期产物(L0-T 的落点);② 本轮按裁定**不提交**的文档与证据脚本。
#
# ⚠️ **「第三方库」这一类现在是空的 —— 而这不是漏了一类**(2026-09-26 实测):
#    整支复核点名要豁免的 `librosa.py` **其实不是一处文件引用**。表里那 5 次出现的
#    **全部**是 `librosa.pyin(...)`(函数调用)—— `librosa.py` 后面紧跟的是 `i`,
#    被本文件正则的 `(?![A-Za-z0-9_])` 这道闸挡住了(实测:`librosa\.py(?![A-Za-z0-9_])`
#    在表里 **0 命中**,而 `librosa\.py` 有 5 命中)。另一处是 `librosa.feature.rms`
#    (`.feature` 不在扩展名名单里)。⟹ 本表**没有任何**第三方文件引用。
#    **机制照旧在**:将来真引到仓外的库时,它以 `unresolved` 的形式红出来,那时在这里
#    加一条并写明理由即可(与「仓库外产物」同一处理)。
EXEMPT = {
    # ① 仓库外的运行期产物(L0-T 的落点,含被试语音 ⟹ 按设计不入仓)
    "transcript.json":
        "运行期产物,落在仓库**外**的 `~/shared/jingxin_recordings/{session_id}/`"
        "(L0-T 的落点:m1-asr 设计 D2:45 / §6.4:128)。含被试语音转写 ⟹ 按设计不入仓。",
    "/transcript.json":
        "同上 —— 它是 `~/shared/jingxin_recordings/{session_id}/transcript.json` 被本钉子"
        "的正则截出来的尾巴,不是另一个文件。",
    "20260926_155559_caf0/transcript.json":
        "同上,具体到 caf0 那一场(2026-09-26 实测存在:`merged.text` 632 字、13 段、"
        "`asr.models` 四键齐全)。",
    "session.json":
        "同上,同目录的会话元数据文件(`asr.models` 等写在这里)。",
    "questions.jsonl":
        "题目台账,落在仓库外的同一目录;`reaction_time` 要与它的 `ask_end` 对得上"
        "(`question_index` 就是连接键)。",

    # ② 本轮按裁定不提交的文档与证据脚本
    "docs/superpowers/plans/2026-09-26-m3-0-l0-column-table.md":
        "本里程碑的**计划文档**。它在 `docs/**` 下,而本轮裁定 `docs/**` 一律不提交"
        "⟹ 盘上有、仓里没有。",
    "docs/superpowers/specs/2026-09-26-l0-column-table-design.md":
        "本里程碑的 **spec**,同上(2026-09-21 那份 spec 已提交,这一份还没有)。",
    ".superpowers/sdd/2026-09-26-m3-0-l0-column-table/task-7-evidence-replay.py":
        "Task 7 的**证据脚本**,按纪律留在 SDD 工作区、不提交(`.superpowers/**` 已 gitignore)。"
        "表里引它是为了指清「那几个数是用哪个脚本量的」。",
    ".superpowers/sdd/2026-09-26-m3-0-l0-column-table/task-final-ts-replay.py":
        "同上(Task 8 / 整支复核那条**喂 `declared_ts`** 的重放脚本)。",
    "data/logs/gesture_emotion_log_20260926_155559_caf0.csv":
        "运行期日志产物,`data/logs/**` 已 gitignore ⟹ 盘上有、仓里没有。表里引它是在指一处"
        "**实测读数**的出处(墙钟重放的值域读数取自这份 CSV)。",
    "data/logs/interview_emotion_log_20260926_155559_caf0.csv":
        "同上(语音侧;`pitch_p90` 那处噪声帧的证据取自它)。",
}


def _strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _refs():
    """表里所有文件引用:`(字段路径, 引用名, 起始行, 终止行)`(行号为 None = 没写行号)。"""
    out = []
    for field, text in _strings(l0.load()):
        for m in _REF.finditer(text):
            a, b = m.group(2), m.group(3)
            out.append((field, m.group(1), int(a) if a else None, int(b) if b else None))
    return out


def _naked_refs():
    """`file:NN / :NN`(或 `file:NN,:NN`)里那个裸号:`(字段路径, 前一个文件名, 裸号起, 裸号止)`。

    这类裸号的**目标不需要猜** —— 就是紧挨在它前面的那个文件(整条一起匹配,归属是捕获出来的)。
    """
    out = []
    for field, text in _strings(l0.load()):
        for m in _NAKED.finditer(text):
            name, a, b, naked = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            lo = int(naked.split("-")[0])
            hi = int(naked.split("-")[1]) if "-" in naked else lo
            out.append((field, name, lo, hi))
    return out


def _tracked_files():
    """仓里**被 git 跟踪**的文件(相对路径)。

    ⚠️ 为什么按 git 而不是按 rglob:仓库根下有一个**未被跟踪的镜像目录**
    `code_data_supplement/`(25MB,2026-07-28 为论文做的模块副本),它让 `app.py` /
    `logger.py` / `config.py` 这些名字在 rglob 下**全都变成不唯一** —— 那会把「唯一性」
    这个判据量成另一个东西(本项目反复栽在"量错了对象")。按 git 跟踪列表判定既确定、
    又与「这份表描述的是**提交进仓的代码**」这个口径一致。
    """
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, f"`git ls-files` 失败({out.returncode}):{out.stderr[:200]}"
    return [f for f in out.stdout.split("\n") if f]


def _resolve(name, tracked, by_basename):
    """引用名 → 仓内命中的文件列表(0/1/多个)。显式路径按后缀匹配,裸名按 basename 匹配。"""
    rel = name[2:] if name.startswith("./") else name
    if "/" in rel:
        return [f for f in tracked if f == rel or f.endswith("/" + rel)]
    return by_basename.get(rel, [])


def _lines_of(rel_path):
    return (REPO / rel_path).read_text(encoding="utf-8", errors="replace").split("\n")


def test_every_file_reference_resolves_to_exactly_one_tracked_file():
    """★ 红法:把表里某处引用改成**名字在仓里不唯一**的裸名(例如 `logger.py` 命中 3 个文件、
    `app.py` 命中 5 个)⟹ 本测试红并逐条点名那几个候选 —— 这正是 Task 5 那次「按 basename
    匹配把三个没碰过的文件的行号一起平移 +7」的机制。"""
    tracked = _tracked_files()
    by_basename = collections.defaultdict(list)
    for f in tracked:
        by_basename[Path(f).name].append(f)

    unresolved = {}
    for field, name, _a, _b in _refs():
        hits = _resolve(name, tracked, by_basename)
        if len(hits) != 1 and name not in EXEMPT:
            unresolved.setdefault(name, (field, hits))

    assert not unresolved, (
        "这些引用的名字在仓里**不是唯一一个文件**(0 个或多个),而它们不在 `EXEMPT` 里:\n  "
        + "\n  ".join(f"{n!r}(首次出现在 {w[0]})→ 候选 {w[1]}" for n, w in sorted(unresolved.items()))
        + "\n—— 要么把引用写成**全路径**(按内容改,不要按名字猜),要么在 `EXEMPT` 里"
          "**逐条写明理由**(第三方库 / 仓库外的产物 / 本轮不提交的文档)。")

    # ③ 豁免表不许闲置:每个键都得**真的**出现在表里,否则就是一张过期白名单。
    seen = {name for _f, name, _a, _b in _refs()}
    dead = sorted(set(EXEMPT) - seen)
    assert not dead, (
        f"`EXEMPT` 里这些名字在表里**根本不出现**了(豁免已经闲置):{dead} —— "
        f"把死掉的豁免删掉,否则它会替将来某个同名引用网开一面。")


def test_every_line_reference_is_in_bounds():
    """★ 红法:把某处行号改成超出那个文件行数的数(例如文件只剩 280 行而引用写 `:9999`)
    ⟹ 本测试红。也覆盖紧随 `file:NN` 之后的裸号(形如 `:605 / :867`)。"""
    tracked = _tracked_files()
    by_basename = collections.defaultdict(list)
    for f in tracked:
        by_basename[Path(f).name].append(f)

    bad = []
    for field, name, a, b in _refs():
        if a is None:
            continue
        hits = _resolve(name, tracked, by_basename)
        if len(hits) != 1:
            if name in EXEMPT:
                # 豁免名:只要它在盘上(相对仓根)就照核界内 —— 这是个**尽力而为**的加码,
                # 文件不在盘上(例如运行期日志被清理过)时跳过,不当成失败。
                cand = REPO / name.lstrip("/")
                if not cand.is_file():
                    continue
                n = len(cand.read_text(encoding="utf-8", errors="replace").split("\n"))
            else:
                continue                      # 唯一性那条已经报过它了
        else:
            n = len(_lines_of(hits[0]))
        if a < 1 or (b or a) > n or (b and a > b):
            bad.append(f"{field}: {name}:{a}" + (f"-{b}" if b else "") + f" —— 该文件只有 {n} 行")

    for field, name, a, b in _naked_refs():
        hits = _resolve(name, tracked, by_basename)
        if len(hits) != 1 or not (REPO / hits[0]).is_file():
            continue
        n = len(_lines_of(hits[0]))
        if a < 1 or b > n or a > b:
            bad.append(f"{field}: {name} 之后的裸号 :{a}-{b} —— 该文件只有 {n} 行")

    assert not bad, ("这些行号落在被引用文件的范围之外(文件变短了、或行号漂了):\n  "
                     + "\n  ".join(bad))

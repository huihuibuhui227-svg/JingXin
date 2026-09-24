# tests/test_asr_log_privacy.py
"""I2(最终全分支审查):识别出的原句不得落进**仓库内**的任何文件。

**失效形态(修复前,已实测):** `voice_interaction/api/app.py` 的 `/asr` 在两处
`logger.info(f"识别结果: '{text}'")`,而 `logging_config.setup_logging()`(app.py
在 import 时调用)给 root logger 装了 `RotatingFileHandler`,落在
**`data/logs/jingxin.log`(检出目录内)**。实测该文件里就有

    2026-09-24 13:07:31 [INFO] voice_interaction.api.app: 识别结果: '我先做了需求分析，然后我们讨论了方案'

即 spec D2 的全部意义(原句**只**落仓库外的
`~/shared/jingxin_recordings/{session_id}/transcript.json`)被这条日志反着走了一遍。
验收判据之一正是 `grep -rn "<原句里的短语>" ~/jingxin → 0` —— 而这句日志正住在那条
grep 会扫到的文件里。

**三条守卫:** ① 接缝写出的日志**内容**里没有原句(变异可证伪);② 跑完这条路径之后
**仓库里没有任何文件**含原句(验收判据的自动化形态);③ 源码级扫描:voice/face/gesture
里不许再有 `logger.*` 调用把"文本变量"当参数 —— 挡住未来重新引入(这一条**在修复前
就是红的**,因为它扫的正是那两句)。
"""

import ast
import contextlib
import logging
from pathlib import Path

import pytest

from voice_interaction.asr.funasr_engine import AsrUtterance
from voice_interaction.asr.transcribe import log_recognition

REPO_ROOT = Path(__file__).resolve().parent.parent

# 一句只在运行时存在的独特短语。**刻意拆成几段再拼**:本文件自身也住在仓库里,
# 若把它写成一个连续字符串,下面那条"仓库内任何文件都不含原句"的扫描会先扫到
# **本文件自己** —— 那样守卫就永远是红的,或者只能靠"排除测试文件"来回避,
# 而排除法会让守卫漏掉一个真实位置。拆开之后,仓库里任何地方出现这句连续短语
# 都只能是"被写进去的"。
_PHRASE_PARTS = ("青柠色的", "纺锤在", "第七码头", "缓慢旋转")
PHRASE = "".join(_PHRASE_PARTS)
assert PHRASE not in Path(__file__).read_text(encoding="utf-8"), "短语没拆干净"

TEXT_VARS = {"text", "recognized_text", "recognized", "answer", "transcript",
             "utterance_text", "sentence", "spoken"}

# 仓库里唯一一个**本来就**解析不了的 .py(未闭合的三引号,预先存在、不在 M1 范围)。
# 显式列出而不是静默跳过:将来若又多出一个,本测试会红,由人来看。
_KNOWN_UNPARSEABLE = {"gesture_analysis/examples/__init__.py"}

# 扫描上限:仓库约 2.6 GB / 1.1 万个文件,其中 1.8 GB 是 328 份生成的 HTML 报告。
# 超过 1 MiB 的文件一律不读(它们是报告产物/图片,不是这条路径的落盘目标)。
# 这条上限是**已知覆盖面损失**,写在 docstring 里而不是藏起来:
# 若将来验收 grep 在大文件里命中,这里也要跟着放大上限。
_SWEEP_MAX_BYTES = 1024 * 1024


def _make_utt(text: str = PHRASE) -> AsrUtterance:
    return AsrUtterance(text=text, n_chars=len(text), n_segments=1, vad_split=False,
                        segments=[])


@contextlib.contextmanager
def _real_logging():
    """按**真实配置**初始化 root logger(`data/logs/jingxin.log`),收尾还原。

    真实配置是关键:本测试要的就是 `setup_logging()` 装的那个 handler 的**形态**
    (接缝的日志要经过它)。收尾把 handler 还原,免得其它测试的日志继续往一个
    已被删掉的临时目录写。
    """
    import logging_config

    root = logging.getLogger()
    before = list(root.handlers)
    logging_config.setup_logging()
    try:
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()
        for h in before:
            root.addHandler(h)


def _flush_root_handlers() -> None:
    for h in logging.getLogger().handlers:
        with contextlib.suppress(Exception):
            h.flush()


def _contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, ValueError):
        return False


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """把 `logging_config` 的落盘目录指到临时目录 —— 只为把落盘内容读回来断言。"""
    import logging_config

    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path)
    with _real_logging():
        yield tmp_path / "jingxin.log"


def test_log_recognition_writes_length_not_the_sentence(isolated_logging):
    """接缝写出的日志**内容**里不得出现原句 —— 连两个字的连续片段都不行。

    红法(见报告里的变异实验):把 `log_recognition` 改成写原句
    (例如 `logger.info("识别结果: '%s'", text)`)→ 本测试立刻红并列出被写进去的片段。
    """
    log_recognition(_make_utt())
    _flush_root_handlers()

    logged = isolated_logging.read_text(encoding="utf-8")
    assert "识别结果" in logged, "接缝根本没写日志,本测试退化为空断言"

    # 逐片段找:连续两个字的片段都不许出现(单个字太容易巧合命中,不作数)
    fragments = {PHRASE[i:i + 2] for i in range(len(PHRASE) - 1)}
    leaked = sorted(f for f in fragments if f in logged)
    assert leaked == [], f"日志里出现了原句的片段:{leaked}\n---\n{logged}"


def test_nothing_under_the_repo_contains_the_recognized_sentence():
    """验收判据的自动化形态:跑完这条路径,`grep -rn "<原句>" ~/jingxin` 必须是 0。

    这条**故意**不把日志目录改到临时目录 —— 它要断言的正是"仓库内那个默认落盘
    位置"上不长出原句。代价是往 `data/logs/jingxin.log`(gitignore 的服务日志,
    生产本来就会追加)写一行长度摘要;那行里只有数字。

    ⚠️ 断言完把这次追加的**尾巴剪掉**(`finally` 里按记录到的旧长度 `truncate`)。
    不剪的话:一旦生产代码把原句写进去,本测试就会红,而**那句话仍留在日志里**
    —— 于是此后每次重跑都红,包括代码已经修好之后(自毒测试)。
    """
    import logging_config

    log_path = Path(logging_config.LOG_DIR) / "jingxin.log"
    size_before = log_path.stat().st_size if log_path.exists() else 0

    try:
        with _real_logging():
            log_recognition(_make_utt())
            _flush_root_handlers()

        hits = []
        for p in REPO_ROOT.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(REPO_ROOT)
            if rel.parts and rel.parts[0] == ".git":
                continue
            try:
                if p.stat().st_size > _SWEEP_MAX_BYTES:
                    continue
            except OSError:
                continue
            if _contains(p, PHRASE):
                hits.append(str(rel))
        assert hits == [], f"仓库里还有文件含识别出的原句:{hits}"
    finally:
        if log_path.exists() and log_path.stat().st_size >= size_before:
            with log_path.open("r+b") as f:
                f.truncate(size_before)


# --------------------------------------------------------------------------
# 源码级守卫:挡住"未来再把原句交给 logger"这条回头路
# --------------------------------------------------------------------------

def _logger_text_leaks(path: Path) -> list:
    """这份源码里有没有 `logger.<级别>(...)` 把文本变量当参数。

    只看 `logger.*` / `logging.*` 的调用:**`print(...)` 不在其内** —— 它只到 stdout,
    不是仓库内的落盘路径(示例脚本与 vendored 客户端的 demo 都靠 print,那条路
    不产生仓库文件)。判据:
      - 参数是 f-string 且里面引用了 `TEXT_VARS` 里的名字 → 命中;
      - 参数**直接**就是 `TEXT_VARS` 里的裸名字 → 命中。
    包一层函数调用不算(`log_recognition(utt)` 与 `len(text)` 都不是把文本交给 logger)。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    leaks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        base = node.func.value
        if not (isinstance(base, ast.Name) and base.id in {"logger", "logging"}):
            continue
        for arg in node.args:
            if isinstance(arg, ast.JoinedStr) and any(
                    isinstance(n, ast.Name) and n.id in TEXT_VARS for n in ast.walk(arg)):
                leaks.append((node.lineno, ast.unparse(node)[:140]))
            elif isinstance(arg, ast.Name) and arg.id in TEXT_VARS:
                leaks.append((node.lineno, ast.unparse(node)[:140]))
    return leaks


def test_no_logger_call_hands_recognized_text_to_the_persistent_log():
    """voice/face/gesture 三包里,`logger.*` 不得再有"把文本变量交给它"的调用。

    这条**在修复前就是红的**:它扫到的正是 `api/app.py` 里那两句
    `logger.info(f"识别结果: '{text}'")`。修复后那两处改走
    `voice_interaction.asr.transcribe.log_recognition(utt)`(只传长度),
    扫描即通过 —— 它守的是"以后别再犯",不是本次修复本身。
    """
    leaks = {}
    unparseable = set()
    for pkg in ("voice_interaction", "face_expression", "gesture_analysis"):
        for src in (REPO_ROOT / pkg).rglob("*.py"):
            rel = str(src.relative_to(REPO_ROOT))
            try:
                found = _logger_text_leaks(src)
            except SyntaxError:
                unparseable.add(rel)          # 解析不了的文件里不可能有合法调用
                continue
            if found:
                leaks[rel] = found

    assert unparseable <= _KNOWN_UNPARSEABLE, (
        f"有新的文件解析不了,扫描出现盲区,需要人来看:{sorted(unparseable - _KNOWN_UNPARSEABLE)}"
    )
    assert leaks == {}, (
        "这些 logger 调用会把文本写进仓库内的服务日志(data/logs/jingxin.log):\n"
        + "\n".join(f"  {f}:{ln} {code}" for f, hits in leaks.items() for ln, code in hits)
    )

# tests/conftest.py
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 把 logger 的**默认**日志目录挪出仓库(会话级,先于任何测试模块 import 生效) ──────
#
# 为什么非在这里不可:导入 `voice_interaction.api.app` 时,模块级就会构造 VoiceLogger,
# 而 `VoiceLogger.__init__` **立刻**写下 CSV 表头(`utils/logger.py` 的 `_write_csv_header`)——
# 不拦的话**每跑一次 pytest 就往仓库 `data/logs/` 里留下一个只有表头的
# `*_log_NONE_<时间戳>.csv`**(2026-09-25 实测:一次两个,面试一个、科研一个;
# 盘上 9/24 那批就是这么攒出来的,先于本轮改动)。这些空文件**不进任何聚合**
# (0 行),但让"仓库目录被测试写脏",也会进报告头的 NONE 桶计数范围。
#
# 单测自己的 `monkeypatch.setattr(logger_module, "LOGS_DIR", …)` 照旧优先(它们在这个
# 会话级值之上再覆盖);这里只是把**构造期**那一次落点挪走。
_LOGS_TMP = Path(tempfile.mkdtemp(prefix="jingxin-test-logs-"))

import voice_interaction.utils.logger as _voice_logger_module  # noqa: E402

_voice_logger_module.LOGS_DIR = str(_LOGS_TMP)

# ── 录音落点也挪出使用者的真实目录(同一理由、同一手法) ──────────────────────
#
# `media_retention.root()` 默认是 `~/shared/jingxin_recordings` —— **使用者真实的数据目录**。
# 2026-09-25 实测:一条没设环境变量的测试往那里写进了 `s1/retention.face.jsonl`
# (文件本身是测试垃圾,但性质是"测试污染真实数据")。单测里各自 `monkeypatch.setenv`
# 只保护那些**记得设**的文件,保护不了后来新加的 —— 所以在这里把默认值兜住:
# 某一个测试若自己设了,它的值优先(在会话级值之上覆盖)。
_RECORDINGS_TMP = Path(tempfile.mkdtemp(prefix="jingxin-test-recordings-"))
# ★ **强制**接管,不用 `setdefault`(2026-10-27 实测踩到)。
#
# 原来写的是 `setdefault`:调用方**已经设了**这个变量时这里就不覆盖 —— 于是
# `JINGXIN_RECORDINGS_DIR=~/JingXin/recordings pytest -q` 会让**整个套件读真素材**。
# 实测症状:`test_html_report_uses_the_l1_engine` 三条全 ERROR
# (`n_valid(8) > n_total(4) —— 把证据说多了`:真素材的语音日志盖过了测试夹具),
# 而单跑这三条是绿的 —— 看上去像代码坏了,其实是测试跑错了对象。
#
# 更阴的是它**反过来咬人**:「手工跑读素材的探针前先 export 这个变量」是对的,
# 同一条习惯照搬到 `pytest` 上却把隔离打穿了。所以这里必须无条件接管:
# 测试套件**永远**不读使用者的真实素材。某个用例要特定的根,自己在用例里
# `monkeypatch.setenv`(它在会话级值之上覆盖,照旧优先)。
os.environ["JINGXIN_RECORDINGS_DIR"] = str(_RECORDINGS_TMP)

# ── 评估流水线的落点同样挪出仓库(第三次、同一理由) ─────────────────────────
#
# `assessment_pipeline.ASSESSMENT_LOG_ROOT` 默认是 `<repo>/data/logs`,而
# `out_dir = ASSESSMENT_LOG_ROOT / kind` **在调用时**查模块全局 ⟹ 改模块属性即可生效。
# 不拦的话:每跑一次全套件往仓库里留 `data/logs/{interview,research}/assessment_note_<ts>.csv`
# (2026-09-26 实测:单跑 `test_media_retention_voice.py::test_research_answer_audio_is_retained`
# 就多 1 个;盘上已积 498 个)。文件被 gitignore ⟹ 进不了提交面,但 `visualize.py`
# **按同一词根扫它给面板列文件** ⟹ 测试噪声出现在报告面板的文件列表里。
# 钉子:`tests/test_log_isolation.py::test_assessment_log_root_is_redirected_out_of_repo`。
# ── 语音帧序列缓存也挪出仓库(第四次、同一理由) ───────────────────────────
#
# `voice_series.series_dir()` 默认是 `<repo>/data/l0_series`。2026-10-24 实测:
# `test_m4_l1_voice_gesture.py::test_missing_voice_log_gap_counts_the_series_that_still_produced`
# 用真 `save_series` 造缓存,却没设 `JINGXIN_SERIES_DIR` ⟹ 往仓库里留下
# `data/l0_series/20261024_010000_ab12.npz`。被 gitignore ⟹ 不进提交面,
# 但它**会让同名的真实场次缓存被顶掉**,而且这套件读的目录就变成了"测试产物 + 真数据"混合。
# 钉子:`tests/test_log_isolation.py::test_series_dir_is_redirected_out_of_repo`。
_SERIES_TMP = Path(tempfile.mkdtemp(prefix="jingxin-test-series-"))
os.environ.setdefault("JINGXIN_SERIES_DIR", str(_SERIES_TMP))

try:
    from voice_interaction.pipeline import assessment_pipeline as _assessment_pipeline

    _assessment_pipeline.ASSESSMENT_LOG_ROOT = Path(
        tempfile.mkdtemp(prefix="jingxin-test-assessment-logs-")
    )
except Exception:  # noqa: BLE001 — 依赖缺失时不让 conftest 整个炸掉
    pass

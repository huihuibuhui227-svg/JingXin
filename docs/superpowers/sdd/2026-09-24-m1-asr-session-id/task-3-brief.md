### Task 3: 三个 logger 写 `session_id`(首列 + 文件名 + NONE)

**Files:**
- Modify: `voice_interaction/utils/logger.py:19,30-44,47-65,77-84`、`face_expression/utils/logger.py:20,31-36,38-60`、`gesture_analysis/utils/logger.py:20,29-38,41-117`
- Test: `tests/test_session_logging.py`, `tests/test_session_id_contract.py`

**Interfaces:**
- Consumes: `voice_interaction.asr.session.NONE_SESSION`
- Produces: 三个 logger 都接受末尾关键字参数 `session_id: Optional[str] = None`,`session_id` 为 CSV **首列**;文件名形如 `{prefix}_{session_id}.csv`,无 id 时 `{prefix}_NONE_{YYYYmmdd_HHMMSS}.csv`

- [ ] **Step 1: 写失败测试(含跨模块常量一致性,文本级读文件、不 import 重模块)**

```python
# tests/test_session_id_contract.py
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "voice_interaction/utils/logger.py",
           ROOT / "face_expression/utils/logger.py",
           ROOT / "gesture_analysis/utils/logger.py"]


def test_none_session_literal_is_identical_everywhere():
    """三处各写一份 NONE —— 值必须一致,否则报告侧的排除规则只对一半生效。"""
    seen = set()
    for src in SOURCES:
        text = src.read_text(encoding="utf-8")
        m = re.search(r'NONE_SESSION\s*=\s*"([^"]+)"', text)
        assert m, f"{src} 缺少 NONE_SESSION 常量"
        seen.add(m.group(1))
    assert seen == {"NONE"}, seen
```

```python
# tests/test_session_logging.py
import csv
from pathlib import Path

from face_expression.utils.logger import DataLogger
from voice_interaction.utils.logger import VoiceLogger


def test_voice_logger_filename_and_first_column(tmp_path):
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path), session_id="20260924_153012_9f3c")
    assert lg.csv_file.name == "interview_emotion_log_20260924_153012_9f3c.csv"
    assert lg.fieldnames[0] == "session_id"
    lg.log_prosody({"pitch_mean": 1.0}, question_index=1, emotion="neutral",
                   feedback="", connective_density=3.5, connective_density_std=0.4, n_rows=4)
    rows = list(csv.DictReader(open(lg.csv_file, encoding="utf-8")))
    assert rows[0]["session_id"] == "20260924_153012_9f3c"
    assert rows[0]["connective_density"] == "3.5"


def test_voice_logger_without_id_writes_none(tmp_path):
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path))
    assert lg.csv_file.name.startswith("interview_emotion_log_NONE_")
    assert lg.fieldnames[0] == "session_id"


def test_face_logger_keeps_session_id_column(tmp_path):
    lg = DataLogger(log_type="video", session_id="20260924_153012_9f3c")
    assert lg.fieldnames[0] == "session_id"
    lg.log_file = str(tmp_path / "face_au_log_20260924_153012_9f3c.csv")   # 调用方会覆盖(现状)
    lg.log({"focus_score": 0.3})
    rows = list(csv.DictReader(open(lg.log_file, encoding="utf-8")))
    assert rows[0]["session_id"] == "20260924_153012_9f3c"
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_logging.py tests/test_session_id_contract.py -q`
Expected: FAIL(文件名仍是时间戳形态 / `fieldnames[0] != "session_id"`)

> ⚠️ **若 `from face_expression.utils.logger import DataLogger` 拉起了 mediapipe/cv2 之类的重依赖而变慢或失败**:不要为了测试去改包的导入结构。改用 `importlib.util.spec_from_file_location` 直接加载那一个文件(它只依赖 `..config`,可在加载前把 `sys.modules["face_expression"]` 设为一个空模块以提供父包),或把 face/gesture 的断言降级为**文本级检查**(读源码断言 `fieldnames` 首项与文件名模板),把行为断言集中在 voice logger 上 —— 三者同构,voice 那份测透即可。

- [ ] **Step 3: 改三个 logger**

`voice_interaction/utils/logger.py`:
```python
NONE_SESSION = "NONE"          # 与 face/gesture 三处同值,由 tests/test_session_id_contract.py 守住

    def __init__(self, log_type: str = 'interview', log_dir: Optional[str] = None,
                 session_id: Optional[str] = None):
        ...
        self.session_id = session_id or NONE_SESSION
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sid_part = self.session_id if session_id else f"{NONE_SESSION}_{timestamp}"
        prefix = 'interview_emotion_log' if log_type == 'interview' else 'research_emotion_log'
        self.csv_file = self.log_dir / f'{prefix}_{sid_part}.csv'
        self.json_file = self.log_dir / f'{prefix}_{sid_part}.json'
        self.fieldnames = ["session_id", "unix_timestamp", "timestamp", ...]          # session_id 置首
        # 末尾追加三列:connective_density, connective_density_std, n_rows
```
`log_prosody(...)` 末尾加三个关键字参数(默认 `None`),写进 `data`。`face_expression/utils/logger.py`:`fieldnames` 首插 `session_id`(`log()` 按 `fieldnames` 过滤,不加就会**静默丢**),构造时用 `session_id` 拼文件名。`gesture_analysis/utils/logger.py`:同样处理(`fieldnames` 首插 + `__init__` 末尾加 `session_id`,构造里用 `log_file_path` 优先、否则用 id 拼名)。

- [ ] **Step 4: 运行,确认通过;提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_logging.py tests/test_session_id_contract.py -q`
Expected: 4 passed

```bash
git add voice_interaction/utils/logger.py face_expression/utils/logger.py gesture_analysis/utils/logger.py tests/test_session_logging.py tests/test_session_id_contract.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): 三份日志写 session_id 首列与带 id 的文件名"
```

---


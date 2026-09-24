### Task 5: face / gesture 收 `session_id`(无 id 时 NONE)

**Files:**
- Modify: `face_expression/api/app.py:129-134,70-74`、`gesture_analysis/api/app.py:137-138,78-83`
- Test: `tests/test_analyze_session_fallback.py`

**Interfaces:**
- Consumes: Task 3 的三个 logger
- Produces: 两个 `/analyze` 在缺 id 时用 `NONE_SESSION`,不再 mint 每请求一个新 uuid

- [ ] **Step 1: 写失败测试**

```python
# tests/test_analyze_session_fallback.py
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_endpoint_mints_a_uuid_session():
    """无 id 时必须是 NONE:旧行为是每请求 uuid4 → 每帧一个新会话、日志文件爆炸。"""
    for rel in ("face_expression/api/app.py", "gesture_analysis/api/app.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "uuid4()" not in text, f"{rel} 仍在 mint uuid 会话"
        assert 'NONE_SESSION' in text, f"{rel} 未使用 NONE_SESSION"
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_analyze_session_fallback.py -q`
Expected: FAIL(仍含 `uuid4()`)

- [ ] **Step 3: 改两个 `/analyze` 与 logger 构造**

```python
# 两个文件都改:
from voice_interaction.utils.logger import NONE_SESSION      # 三处同值(测试守住)
...
    session_id = session_id or NONE_SESSION
```
logger 构造处把 `session_id` 传进去,文件名用会话 id(face:`log_path = os.path.join(LOGS_DIR, f'face_au_log_{session_id}.csv')`;gesture 同理传 `log_file_path`)。

- [ ] **Step 4: 运行,确认通过;提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_analyze_session_fallback.py -q`
Expected: 1 passed

```bash
git add face_expression/api/app.py gesture_analysis/api/app.py tests/test_analyze_session_fallback.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): face/gesture 接受 session_id,无 id 写 NONE(不再每请求 mint uuid)"
```

---


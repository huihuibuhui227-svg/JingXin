### Task 4: voice 端点接线 + 拆掉 vosk

**Files:**
- Modify: `voice_interaction/api/app.py:16,50-61,116-220,225-236,266-296`、`voice_interaction/pipeline/speech_recognition_pipeline.py:13,20-43,73-110,149`、`voice_interaction/__init__.py:40`
- Test: `tests/test_voice_transcribe_helper.py`
- Delete: `vosk-model-cn-0.22/`(2.0 GB,未被 git 跟踪)

**Interfaces:**
- Consumes: Task 1–3 的全部产物
- Produces: `voice_interaction.api.app._transcribe(audio_data: bytes) -> AsrUtterance`(模块级 `asr_engine` 可替换,便于测试与验收)

- [ ] **Step 1: 写失败测试(只测 helper,不 import 整个 app)**

```python
# tests/test_voice_transcribe_helper.py
import importlib

from voice_interaction.asr.funasr_engine import AsrUtterance


def test_transcribe_helper_delegates_to_engine(monkeypatch):
    """_transcribe 只做转发:引擎抛错就抛错,识别为空就返回空文本(不写 0)。"""
    mod = importlib.import_module("voice_interaction.api.app")
    calls = {}

    class FakeEngine:
        def transcribe_pcm(self, pcm):
            calls["pcm"] = pcm
            return AsrUtterance(text="然后我们说", n_chars=5, n_segments=1, vad_split=False, segments=[])

    monkeypatch.setattr(mod, "asr_engine", FakeEngine())
    utt = mod._transcribe(b"\x00" * 320)
    assert calls["pcm"] == b"\x00" * 320
    assert utt.text == "然后我们说"
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_voice_transcribe_helper.py -q`
Expected: FAIL(`AttributeError: module ... has no attribute '_transcribe'`)

- [ ] **Step 3: 改 `api/app.py`**

```python
# 顶部:删掉 from vosk import Model, KaldiRecognizer 与 MODEL_PATH/vosk_model 构造
from voice_interaction.asr.funasr_engine import AsrUtterance, FunASREngine
from voice_interaction.asr import session as session_mod, transcript_store
from voice_interaction.asr.connective_density import connective_density

asr_engine = FunASREngine()


def _transcribe(audio_data: bytes) -> AsrUtterance:
    """把音频交给 ASR 引擎。异常上抛(由调用方转成 HTTP 错误),不在这里吞。"""
    return asr_engine.transcribe_pcm(audio_data)
```

四处识别点(`:144`、`:203`、`:281`、`:414`)替换为:
```python
        utt = _transcribe(audio_data)
        text = utt.text.strip()
```

`/interview/start` 发号:
```python
@app.post("/interview/start")
async def start_interview():
    try:
        interview_assessment.reset()
        first_question = interview_assessment.get_next_question()
        if not first_question:
            raise HTTPException(status_code=500, detail="无法获取问题")
        sid = session_mod.new_session_id()
        transcript_store.ensure_manifest(sid, _asr_meta())
        tts_engine.speak(first_question)
        return {"status": "started", "session_id": sid, "question": first_question}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动面试失败: {str(e)}")
```
`_asr_meta()` 返回 `{"engine": "funasr", "endpoint": f"ws://{asr_engine.host}:{asr_engine.port}", "models": {...}, "asr_confidence": None, "asr_confidence_source": "unavailable"}`。

`/interview/answer_audio` 收 id + 落转写 + 写密度:
```python
@app.post("/interview/answer_audio")
async def submit_answer_audio(audio: UploadFile = File(...), session_id: str = None):
    sid = session_id or session_mod.NONE_SESSION
    ...
        utt = _transcribe(audio_data)
        text = utt.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="未识别到有效语音")
        transcript_store.append_utterance(sid, utt)                     # 仓库外
        density = connective_density(text)                              # 文本层,过短则 None
        voice_logger.session_id = sid                                   # 首列随会话
        voice_logger.log_prosody({}, question_index=..., emotion="", feedback="",
                                 connective_density=density, connective_density_std=0.0, n_rows=1)
        interview_assessment.add_answer(text)
        ...
        return {"status": "success", "session_id": sid, "recognized_text": text,
                "connective_density": density}
```
`/asr` 同样加 `session_id: str = None` 参数,识别后 `transcript_store.append_utterance(sid, utt)`(纯 ASR 不写特征行)。

`speech_recognition_pipeline.py`:删掉 `from vosk import Model, KaldiRecognizer`、`MODEL_PATH` 与模块级 `RuntimeError`;`__init__` 里 `self.model = Model(...)` 换成 `self.engine = FunASREngine()`;两处识别改走 `self.engine.transcribe_pcm(...)`,返回构造保持不变(`SpeechRecognitionResult(text=..., confidence=..., is_final=True, audio_data=audio_obj)`)。`voice_interaction/__init__.py` 不再有缺模型即崩的路径。

- [ ] **Step 4: 运行,确认通过 + 全仓无 vosk 残留**

```bash
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_voice_transcribe_helper.py -q          # 1 passed
grep -rn "vosk\|KaldiRecognizer" voice_interaction/ report_frontend/ --include="*.py" | wc -l     # 期望 0
~/miniconda3/envs/jingxin/bin/python -c "import voice_interaction; print('import ok')"
```

- [ ] **Step 5: 删模型与代码,提交**

```bash
rm -rf vosk-model-cn-0.22/            # 2.0 GB,未被跟踪;代码已无引用
git add voice_interaction/api/app.py voice_interaction/pipeline/speech_recognition_pipeline.py voice_interaction/__init__.py tests/test_voice_transcribe_helper.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): voice 换 FunASR(/interview/start 发号、落转写、写连接词密度),删除 vosk 代码与 2 GB 模型"
```

---


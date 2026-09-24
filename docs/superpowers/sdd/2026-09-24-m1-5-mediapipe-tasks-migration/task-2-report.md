# Task 2 报告:face 适配(mediapipe tasks 迁移)

**状态**:DONE_WITH_CONCERNS
**提交**:`db3cca156761379649e06697641dbe3b4132857a`(`feat(m1.5): face 接入 tasks 探测器(按会话 + 帧计数 + TTL 释放)`)
**改动文件**:4 个 —— `face_expression/pipeline/video_pipeline.py`、`face_expression/api/app.py`、
`tests/test_face_detector_wiring.py`(新)、`tests/test_analyze_session_fallback.py`(**超出任务书文件清单,见 §2**)

---

## 1. 逐步记录

### Step 1:写失败测试
创建 `tests/test_face_detector_wiring.py`,逐字照抄任务书 Step 1 的三段代码(含 `_FakeDetector`
与 `_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)`)。

### Step 2:确认红
```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_face_detector_wiring.py -q
3 failed in 0.21s

E   TypeError: VideoPipeline.__init__() got an unexpected keyword argument 'detector'
tests/test_face_detector_wiring.py:52: TypeError
```
3 条全红,报错文本与任务书 Expected 一字不差。

### Step 3:改 `video_pipeline.py`
- 删掉 `self._face_mesh = None`(原 :24)与整个 `face_mesh` 惰性属性(原 :31-42)。
- `__init__` 增加 `detector=None` 关键字参数,缺省时 `from .detector import FaceDetector` +
  `from ..config import FACE_MODEL` 自造一个真的,赋给 `self.detector`。
- `process_frame` 开头换成 `landmarks_norm = self.detector.detect(image_rgb)`,判空条件改为
  `if not landmarks_norm:`。从 `nose_tip = np.array(landmarks_norm[1])` 起整段几何层原样未动。
- 新增 `reset()` / `close()`,各自转发给 `self.detector`。

### Step 4:确认绿
```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_face_detector_wiring.py -q
3 passed in 0.19s
```

### Step 5:改 `app.py`

**(a) TTL 回收**(`get_or_create_pipeline` 内):`del session_pipelines[sid]` → 
`pipeline, _ = session_pipelines.pop(sid)` + `pipeline.close()`,照抄任务书。

**(b) `/session/{session_id}/reset`** —— 见 §2,采纳裁决:`close()` 后删,**不调 `reset()`**。

**(c) `__main__` 自检探针**:整段 `mp.solutions` 兼容性探针(含它那个吞异常的
`try/except Exception`)整体换成 `verify_models()`,**不吞异常**;并删掉 :20 的模块级
`import mediapipe as mp`。

### Step 6:全量
```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest -q
189 passed in 10.82s
```
(HEAD 基线 = 186 passed / 0 failed,实测见 §3;+3 条新测 = 189。)

### Step 7:实跑验收

实跑脚本 `/tmp/t2_step7.sh`,与任务书一致,唯一改动是**把 `sleep 8` 换成就绪轮询**
(最多 30 s 轮询 `/health`,拿到 200 才继续)—— `sleep` 在前台被本环境拦,且轮询比定长
睡眠更确定。其余命令原样。

```
server ready after 13 polls
SID=t_20260924_211132_aaaa
HTTP_CODE: 200
```
- **HTTP 状态码:200**(迁移前这里必然 500)。
- **落盘文件**:`/home/huihuibuhui/jingxin/data/logs/face_au_log_t_20260924_211132_aaaa.csv`,
  811 bytes,首列为 `session_id` 表头 + `t_20260924_211132_aaaa`,文件名带会话 id。
- 响应体 `status":"success"`,`session_id` 与请求一致,AU/情绪/紧张度/时序统计字段齐全
  (`head_yaw`:0.314,`avg_ear`:0.414 等真实数值,非默认回落值)。
- 服务端日志确认真跑了 tasks 图(`face_landmarker_graph.cc` / `inference_feedback_manager.cc`
  的 native 警告),且自检通过:`正在自检人脸模型...` → `人脸模型自检通过`(耗时约 5 s,
  即模型加载)。`data/logs/` 已被 `.gitignore:45` 覆盖,不会污染仓库。

**额外补跑(任务书未要求,为验证 §2 的裁决)**:`/tmp/t2_reset_check.sh`
```
frame1            -> 200
summary           -> 200 (frames before reset: 2)
reset             -> 200 {"status":"success","message":"会话 t_20260924_211156_rst 已重置"}
summary after reset -> 404   (会话确实被删掉,不是"保留但清空")
frame2 (rebuild)  -> 200     服务端日志再次 "创建新会话: t_20260924_211156_rst"
```
即"删掉 → 下次请求重建全新管线(含全新探测器)"这条语义在真实端点上成立。
(`frame_count: 2` 而非 1 是**迁移前就有的**行为:`process_frame` 在历史为空时会先注入一帧
零帧,原 :89-107 的逻辑,与本次改动无关。)

### Step 8:提交
`git add` 任务书那三个文件 + `tests/test_analyze_session_fallback.py`(理由见 §2)。
`docs/superpowers/plans/2026-09-24-m1-5-mediapipe-tasks-migration.md` 的改动**在我开工前就
存在**(初始 `git status` 即为 ` M`),不是我改的,已**保持未提交**。

---

## 2. Step 5(b) 的裁决 + 两处必要的偏离

### (1) Step 5(b):采纳"删掉整个会话"语义 → 只 `close()`,不 `reset()`

读到的实际形状(`app.py:351-359`,改动前)确实是 `del session_pipelines[session_id]`
+ `session_loggers.pop(...)`,**不是**"重置状态但保留会话"。因此:

```python
pipeline, _ = session_pipelines.pop(session_id)
pipeline.close()          # 删之前先放掉探测器的 native 句柄(spec §6.3)
session_loggers.pop(session_id, None)
```

**未**在该函数里调用 `pipeline.reset()` —— 会话整个被删,下一次请求走
`get_or_create_pipeline` 重建 `VideoPipeline`,连带一个全新 `FaceDetector`(`_frame_index = 0`),
帧计数天然从 0 开始,`reset()` 是多余的。`VideoPipeline.reset()` 方法**保留**(有契约测钉着,
且若日后端点语义改成"保留会话"就用得上)。TTL 回收那个循环同理,一律 `close()` 然后删。

我把这条语义写进了端点的 docstring,免得下一个读代码的人以为漏调了 `reset()`。

### (2) 偏离一:`process_frame` 的返回值第二槽(任务书此处有 bug)

任务书 Step 3 说"从 `nose_tip` 那一行起**一个字都不改**",但原 :187 是:

```python
return result, results, result.to_dict()
```

`results` 正是被删掉的那个局部变量(`results = self.face_mesh.process(...)`)。**字面照抄会
在每次检出人脸时 `NameError`** —— 这是任务书里的一处真实缺口。

我改成 `return result, landmarks_norm, result.to_dict()`,理由:
- `landmarks_norm` 就是 `results` 在这条线上的**直接后继**,保留三元组形状(两处解包点
  `app.py:226` 与 `examples/run_video_analyzer.py:110` 都依赖 arity 3)。
- 真值语义与迁移前一致(有真值 ⟺ 本帧检出脸);若改传 `None`,老消费方 `if results:` 会把
  "检出脸"读成"没检出",属于本项目一贯拒绝的静默错误。
- spec §89 已确认原始 results 对象在 `api/app.py:226` 绑给 `mesh_results` 后**再没被读过**,
  所以这一槽今天没有活的读取方。
- `face_expression/examples/run_video_analyzer.py:110` 会 `results.multi_face_landmarks`
  —— 该脚本**本任务文件清单之外**,且在本环境**已经跑不起来**(mediapipe 1.0.0 无
  `mp.solutions`,`mp_drawing` 恒为 `None`)。**我没动它**,仅在此标记。

### (3) 偏离二:`tests/test_analyze_session_fallback.py` 的假替身(**超出任务书清单**)

Step 5(a) 让 TTL 回收调用 `pipeline.close()`,而该测试文件的 `_FakeFacePipeline` 替身
(定义于 :128)是 `VideoPipeline` 的手写替身,**没有 `close()`**。它的时钟每次推进 1000 秒
(> `SESSION_TIMEOUT=300`),第二帧**必然**走 TTL 回收那条路,于是:

```
AttributeError: '_FakeFacePipeline' object has no attribute 'close'
  face_expression/api/app.py:126  in get_or_create_pipeline
```

→ `test_face_no_id_writes_none_and_reuses_one_file` 红。这是**我的改动引入的真实回归**,
不是既有红。

我**必须**改这个文件才能满足 Step 6 的"0 红"。改动只有 3 行:给替身加 `closed` 标志与
`close()`(带 docstring 说明为什么)。**没有**动任何被测逻辑。之所以不选"让 `app.py` 用
`getattr(pipeline, 'close', lambda: None)()` 兜底":那正是本项目一贯拒绝的静默失败,
而且会掩盖"真探测器没被关"这类缺陷。

**这一条超出任务书文件清单("只改任务书里列出的文件"),请审查者重点确认是否接受。**

---

## 3. 基线澄清:不存在"既有红"

我一度把 `tests/test_assert_coverage.py::test_no_assert_is_dead` 误判为既有失败,**这个判断是错的**,
在此更正:该测试会在**进程内**跑一遍整个套件并断言 `inner_exit_code == 0`。我第一次测基线时只
`git stash` 了实现文件、**没挪走新写的红测**,于是内层套件因为那 3 条红测而返回非零 → 它跟着红。

严格基线(实现与红测同时移开,HEAD `a29d249`):
```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest -q     # 实现 + 新测均移开后
186 passed in 10.53s
```
**HEAD 是干净的 186 passed / 0 failed,没有任何既有红**。移回后 189 passed / 0 failed。
交付树是绿的。

---

## 4. 可疑但没动的东西

1. **`examples/run_video_analyzer.py:110-113`** —— 见 §2(2)。它的 `results.multi_face_landmarks`
   在本次改动后语义不成立;该脚本在本环境本就跑不起来(无 `mp.solutions`)。不在任务书清单内,
   未动。
2. **`face_expression/examples/` 下其他脚本**同样可能仍引用 `face_mesh` 属性 —— 未逐一排查,
   亦未动。`grep face_mesh face_expression/`(排除 `examples/`)已确认**活路径零残留**。
3. **`code_data_supplement/` 是仓库内另一份 face/gesture 副本**,仍含 `MEDIAPIPE_CONFIG` 与
   `mp.solutions`。看起来是历史快照/留档而非活代码,未动。
4. **`gesture_analysis/**` 一个字未碰**(包括 `gesture_analysis/config.py` 的 `MEDIAPIPE_CONFIG`
   与 `gesture_analysis/api/app.py` 的 `mp.solutions`)——按指示留给另一任务。
5. **`face_expression/config.py` 未动**;其中 `MEDIAPIPE_CONFIG` 现在只有一行注释说明它是死代码,
   那是 Task 1 的产物。
6. **TTL 回收的 `close()` 只被"执行到"、未被"断言到"**:`test_analyze_session_fallback.py` 会真的
   走到那行(没有 `close()` 就 AttributeError),但我没有为它加 `assert fake.closed is True`
   —— 加了就更要动那个测试文件。替身里我留了 `closed` 标志,将来要加断言可以直接用。
7. **Step 7 我把 `sleep 8` 换成了就绪轮询**(环境拦前台 `sleep`),这是与任务书唯一的命令差异,
   已在 §1 标注。

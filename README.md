# 《衡才智析——多模态行为证据分析系统》（原开发代号 JingXin）

> **软著名**：`衡才智析——科研行为证据门控评估系统 V1.0` ｜ **核心理念**：「不给分，只给证据与缺口」 / 「仪器可以是黑盒，裁判必须是白盒」

一套**多模态行为证据披露**系统。把一场面试或科研汇报中受试者的面部动作单元、3D 头姿与 52 维表情系数、上肢骨骼运动、语音声学韵律与转写话语结构采下来，落成逐帧/逐回答的 CSV，经由姿态残差校准与多通道动力学特征层，最后交给**不打分的白盒证据门与可解释机器学习裁判层（`GaussianHMM + PELT + Split Conformal`）**按有出处的判据披露「哪些行为线索有证据、处于哪类证据核验阶段、哪些没量到且为什么、出自哪套仪器」。

> ⚠️ **它不是心理测量工具，也不做心理识别结论。** 这条是硬约束，不是措辞偏好：
>
> - 报告层**不渲染任何复合分、不给档位** —— 它自己写着「尚无标定样本，不同指标的量纲无法折算为同一尺度」。
> - 判定阈值/满量程**只能来自物理下限或自采数据**。上游数据集（RecruitView）的许可禁止把在它的分布上拟合出来的参数用于真实招聘/筛选/心理画像 —— **在数据集分布上拟合的参数同样属禁**。
> - 采集自评量表**必须先过科技伦理审查**（这一条拦住整个 L2 层）。
>
> 所以改这个系统的第一原则不是「让它多出分」，而是**「别让它说不成立的话」**。

本文面向**接手的人**，只讲「东西在哪、怎么起、怎么查」。
逐层细节与所有权威出处见 `docs/开发者文档.md`；任务清单的唯一权威是 `l0_columns.json`。

---

## 0. 目录布局（先看这个）

工作区根 `/home/zgy/JingXin/` **不是 git 仓库**（所以别往根上放文件，没人管）：

| 路径 | 是什么 | 版本控制 |
|---|---|---|
| `jingxin/` | 本仓库：后端 + 报告层 + 工具 | ✅ git |
| `frontend/` | **另一个仓库**：React + Vite + TS 前端 | ✅ git（独立） |
| `recordings/` | 40+ 场素材（视频/帧/留存账本）。**纯数据** | ❌ 不入库 |
| `.env` | `JX_ADMIN_PASSWORD`。**故意留在仓库外** | ❌ |

仓库内布局：

```
app.py                  Flask 报告面板(:5000，含脱敏/鉴权、健康探活、scene_profiles、Career MCP、纵向对比/随访、人工复核注记、本地辅助解读、题库检索与科研基准)
redline_guard.py        品牌与法规红线词/打分正则硬守卫（独立于 report_frontend/）
mcp_career/             标准 JSON-RPC 2.0 Career MCP Server（4 工具 / 10 岗位，只披露证据覆盖与缺口，零匹配分）
face_expression/        面部服务(:8000)   内存 cv2.imdecode，输出 iris + AU + 52 Blendshape + 3D 姿态 + 画质门控（98 列 CSV），支持无影像隐私拦截
gesture_analysis/       手势服务(:8002)   rtmlib RTMW 133 点全身双手引擎（兼容 Hand 21 / Pose 33 口径，59 列 CSV），支持无影像隐私拦截
voice_interaction/      语音服务(:8001)   Parselmouth 声学 + Silero VAD + CAM++ + FunASR 转写 + faster-whisper 词级核验，支持多模块题库与无影像拦截
report_frontend/        报告层:data_loader → 回答/三阶段窗口 → L1(41 项)+L2(姿态校准/动力学/节律)+L3(HMM/PELT/Split Conformal)
                         → scene_profiles / scene_extensions / longitudinal_tracker / reviewer_notes / report_generator 单文件 HTML
                         （旧引擎 feature_engine → research_mapper 仅 ?engine=legacy 对照）
research_lab/           严格隔离的离线多语料科研基准实验室（MIT / RecruitView / ChaLearn V2 / 中英迁移，绝不参与报告计分）
models/                 MediaPipe 模型 + signal_quality_clf.joblib + behavior_regime_model.joblib + referee_hmm/conformal_gate.joblib
media_retention.py      留存与账本(三服务共用，支持 full / audio_only / anonymous_kinematics_only 隐私拦截)
l0_columns.json         ★ L0 列清单的唯一权威(status × maturity 两轴，含面部 98 列契约)
tools/                  起停(start_all.sh / stop_all.sh) / 自检 / 基准 / 3D 回填 / 裁判重训 / 转写
docs/                   文档(见 §4「权威在哪」)
data/logs/              运行日志 CSV(gitignored)
data/transcripts/       ASR 转写产物(gitignored,**含真实受试者个人信息,只能留服务器**)
```

---

## 1. ⚠️ 换机器必改的四处

**这一节是本文件最有用的部分。** 本仓反复踩同一个坑：路径写死 → 换台机器就静默走错
（`jx_env.sh`、`start_all.sh`、`jx_check_session.py` 都中过）。**路径一律按候选探测，可用环境变量覆盖。**

| 项 | 本机实际值 | 覆盖变量 | 候选顺序 |
|---|---|---|---|
| Python 解释器 | `~/anaconda3/envs/jingxin/bin/python` | `JX_PY` | anaconda3 → miniconda3 |
| 仓库根 | `~/JingXin/jingxin` | `JX_REPO` | `~/JingXin/jingxin` → `~/jingxin` |
| 素材根 | `~/JingXin/recordings` | `JX_RECORDINGS` | `~/JingXin/recordings` → `~/shared/jingxin_recordings` |
| 前端仓库 | `~/JingXin/frontend` | — | `~/JingXin/frontend` → `~/JingXin-frontend` |

> 解释器**不可用** `~/huihui/bin/python`（数据科学环境，本仓依赖没装）。
>
> 探测逻辑的实现在 `tools/jx_env.sh`，新脚本请 `source` 它，不要另写一套。

---

## 2. 五分钟跑起来

```bash
cd ~/JingXin/jingxin
source tools/jx_env.sh          # 拿到 $JX_PY / $JX_REPO / $JINGXIN_RECORDINGS_DIR

# 生产模式一键启动(三个采集服务 + 面板 + 前端,幂等,自动加载 ~/JingXin/.env + HTTP 探活)
bash tools/start_all.sh --prod

# 一键优雅停止全部 5 个服务(兼容 systemd --user 与手动拉起)
bash tools/stop_all.sh
```

| 服务 | 端口 | 作用 |
|---|---|---|
| face_expression | 8000 | 面部 478 点 AU / iris / 52 Blendshape / 原生 3D 姿态角 / 画质门控 |
| voice_interaction | 8001 | Parselmouth 声学 / Silero VAD / FunASR 转写 / faster-whisper 词级核验 |
| gesture_analysis | 8002 | rtmlib RTMW 133 点全身与双手骨骼 |
| app.py（面板） | 5000 | 报告面板 API / 脱敏输出 / 导出 / 题库 / 场景引擎 / Career MCP / 纵向追踪 / 复核注记 / 本地助手 / 科研基准 |
| 前端（Vite） | 5173 | 《衡才智析》Web 界面 |
| FunASR 运行时 | 10095 | 独立部署，见 §5 |
| 本地 LLM（可选） | 8081 | `Qwen2.5-14B` OpenAI 兼容接口（仅按需摘要/追问，零评分参与） |

一键全链路自检：

```bash
curl --noproxy '*' -s http://127.0.0.1:5000/api/system/health
```

> ⚠️ **三个服务没有启动脚本，必须 `setsid` 脱离终端起。**
> 否则终端一 `Ctrl-Z` 就成 T 态 —— 端口在听却不回话，很难查。`start_all.sh` 已经替你做了。

---

## 3. 常用命令

```bash
PY=$JX_PY   # 或写全 ~/anaconda3/envs/jingxin/bin/python

# ── ★ 改完东西先跑这个（引用/登记/口径/变异/提交范围，约 4 秒）────
$PY tools/jx_selfcheck.py                 # 元检查（不含全量）
$PY tools/jx_selfcheck.py --all           # 追加全量套件（70~100 秒）
$PY tools/jx_selfcheck.py --mutations     # 追加：真跑变异登记表（每条一个子进程）
# 它管的是「我上一次改的东西有没有把别处说过的话弄成假话」——
# 引用指不指得着、维度死权重对不对、docstring 里的「红法」是不是真的。
# ⚠️ 在本沙箱里 --all 会因 test_recordings_browser（禁止跨设备目录改名）报 ✗。

# ── 测试 ──────────────────────────────────────────────
# ★ 全量门要带 JX_RUN_MUTATIONS=1:覆盖率守卫拒绝给「被跳过」的测试背书,
#   而变异实测默认是跳过的(每条一个子进程,约多花 30 秒)。
JX_RUN_MUTATIONS=1 $PY -m pytest -q    # 全量门(期望全绿)
$PY -m pytest -q                       # 快跑:会**照实**报 1 红(test_no_assert_is_dead)
NUMBA_CACHE_DIR=/tmp/numba-cache $PY -m pytest -q   # numba 需要可写缓存目录

# ── 结构图与清单（脚本生成，不会过期）────────────────
$PY tools/jx_inventory.py              # 重刷 docs/资源总览.md
$PY tools/jx_inventory.py --check      # 只比对；过期退出码 1

# ── 审计一场录制能不能进后续流程 ─────────────────────
$PY tools/jx_check_session.py --all
$PY tools/jx_check_session.py 20260929_195403_040b

# ── 拿留存视频的音轨走 ASR 采语言文本（幂等）────────
$PY tools/jx_transcribe_recordings.py --limit 1   # 先试一场
$PY tools/jx_transcribe_recordings.py            # 全部
# 产物 data/transcripts/，已 gitignore

# ── 合并门 / 回归防线 ────────────────────────────────
$PY experiments/duration_audit/reaggregate_normalized.py --verify-legacy   # 期望 0 / 2835510
```

---

## 4. 权威在哪（只记这一张表，别抄第二份）

| 要看什么 | 去哪 |
|---|---|
| **现在该做什么**（甲可立即开工 / 乙卡外部条件 / 丙结构性 / 丁） | `docs/下一步.md` §一 |
| **任务清单的唯一权威**（86 列，两轴） | `l0_columns.json` + `l0_columns.py` |
| 里程碑进度（M1→M5） | `docs/下一步.md` §五 |
| **必须带进后续轮次的经验**（14 条，踩出来的） | `docs/下一步.md` §二 |
| 全盘结构 / 计数（脚本生成） | `docs/资源总览.md` |
| 系统是什么/不是什么、逐层详解、已知陷阱 | `docs/开发者文档.md` |
| 工具说明（起服务 / 验收 / 录制） | `tools/README.md` |
| 素材体检结论 | `docs/素材体检_2026-10-03.md` |
| 工作区版本梳理 | `docs/版本梳理_2026-10-03.md` |
| 过程记录（某个决定当时怎么来的） | `docs/复盘/2026-09-25_09-27_会话日志.md` |

**三条纪律**（也是踩出来的）：
1. 说「验证过」之前先问：**我跑的这条会不会执行到我要证的那段代码？**
2. **散文里的数会静默过期** —— 数只说一次，能算的就别手写。
3. **重放必须喂留存账本的 `declared_ts`**；用墙钟会得出「推理不可复现」的**假**结论。

---

## 5. ASR 与说话人

项目通过 `voice_interaction/asr/` 这一层调本机 FunASR 运行时（`ws://<host>:10095`）。
引擎与端口在 `voice_interaction/asr/asr_config.json`。

⚠️ **本部署的 ASR 返回里 `spk_name` 恒为 `'unknown'`。** 原因是服务端脚本
`~/huihuibui/FunASR/runtime/python/websocket/funasr_wss_server.py` 里：

- 声纹模型 `iic/speech_campplus_sv_zh-cn_16k-common` 是**硬编码加载的**（约 285 行），
  所以**不需要为此重启服务**；
- 但它要做**声纹辨认（1:N）**，依赖同目录下的 `speaker_db.json` 注册库，
  而**该文件不存在** ⟹ 没有可辨认的对象，只能回 `'unknown'`。

想要说话人信息有两条路，见 `docs/素材体检_2026-10-03.md` §4 与本文 §6：

- **A. 服务端声纹辨认**：建 `speaker_db.json` 注册参考人声 ⟹ 返回"是不是这个人"。
  适合已知说话人（如固定的老师）。
- **B. 离线话者分离**：用 `~/anaconda3/envs/funasr` 里的 `funasr` 离线 pipeline
  （CAM++ 嵌入 + 聚类），**无需注册**，输出"谁在什么时候说了什么"。
  适合"老师 vs 学生"这种未知两人。

---

## 6. 已知坑（都别重踩）

- **换机器路径写死** → 见 §1。这是本仓最高频的坑。
- **`~/.claude` 之外散着约 300 处 `文件:行` 引用，零钉子覆盖**；引用代码请写**符号名**，不要抄代码。
- **`media_retention.py` 的 docstring 里那句「全仓无 `VideoWriter`」是历史**，别当现状。
- **审计工具自己的路径漂移比不审计更坏** —— 它会给你一批看起来很像真的的假失败。
- **`code_data_supplement/` 是冻结的论文复现包**，是本仓三个服务的**旧版分叉**，**一个字节都不许动**。
- 正式素材与试跑**只能靠目录位置区分**（`meta.json` 同机同脚手架，分不出来）。

---

## 7. 提交纪律

```bash
git add <具体路径>        # ★ 禁止 git add -A
```

`.gitignore` 里那句警告是有来历的：`code_data_supplement/` 曾被 `git add -A` 卷进过一次提交
（2026-09-27，已回退）。
另外 `data/transcripts/` 与 `data/logs/`、`data/output/` 都已忽略 —— **前者含真实受试者个人信息**。

---

> **最后一句**：这个系统的价值在于**它说的每句话都有依据**。
> 改它的第一原则不是「让它多出分」，而是「别让它说不成立的话」。


---

## 2026-10-26 新增（本轮整改）

### 新模块

| 文件 | 作用 |
|---|---|
| `question_bank.py` | **多套题库**（面试/科研），每模块可存多套、选一套"当前生效"；参考答案可选（只存不用） |
| `exports.py` | 数据导出：三份 CSV、转写 TXT/CSV/JSON、按模块 PNG 图表（matplotlib，列动态取） |
| `face_expression` / `gesture_analysis` 的 `/analyze?probe=1` | 设备自检探针：**只检测，不落帧、不写 CSV、不落取景决策** |

### 新接口

| 接口 | 说明 |
|---|---|
| `GET /api/question-bank/<module>` | 全部题库 + 当前生效那套 |
| `POST /api/question-bank/<module>/banks` | 新建一套（不覆盖已有） |
| `PUT/DELETE /api/question-bank/<module>/banks/<id>` | 改/删某一套 |
| `POST /api/question-bank/<module>/active` | `{id}` 设为当前；`null` = 内置默认 |
| `GET /api/question-bank/sample/<module>` | 下载随身示例题库（8 题 + 参考答案） |
| `GET /api/export/sessions` | 会话列表 + 姓名/学号/院系/场次 |
| `GET /api/export/<sid>/files` | 可导出清单（CSV/转写/行数） |
| `GET /api/export/<sid>/csv/<key>` | 下载 face/gesture/voice_* CSV |
| `GET /api/export/<sid>/transcript?format=txt\|csv\|json` | 转写导出（含题干/姓名） |
| `GET /api/export/<sid>/chart?module=face\|gesture\|voice` | PNG 图表 |

### 报告与面板

- 报告头：被测者/学号/院系/场次 + **测试时间**（铸号时刻）与**报告生成时间**；
- 文件名带姓名（`...-候选Y.html`），同一秒撞名不覆盖（追加会话号）；
- 面板 `app.run(debug=False)`（`JX_PANEL_DEBUG=1` 才开）—— debug reloader 会清空内存任务表；
- 报告按钮的状态判断改为服务端实际值 `success/error`。

### 采集端

- 评估页在**铸号之前**做设备自检：面部 `face_scale ≥ 0.20` + 至少一只手 + 麦克风；
  探针不落盘；未通过不铸号（前端 `PreflightGate.tsx`）。

### 启动注意（本机）

- 用 `tailscale serve` 发布时，**语音/面部/手势服务必须 `JX_BIND=127.0.0.1`**，
  否则会占 tailscale IP，`tailscaled` 的 TLS 监听起不来（静默重试）。
- 本机语音服务导入 librosa/numba 需要可写缓存：`NUMBA_CACHE_DIR=/tmp/numba_cache`
  （`start_all.sh` 之外手动起服务时要带上）。
- 手动重启面板要带 `CORS_ORIGINS`（含 `https://gpuserver.tail4ffce2.ts.net:5173`）

### 新增测试

`tests/test_question_bank.py`（多套/切换/删除/旧格式迁移）、`tests/test_exports.py`
（清单/转写/图表/无数据）、`tests/test_report_subject_block.py`（身份块/两个时间/文件名/撞名）。

### 已知

- 整场 `camera.webm` 上传实测约 0.5 MB/s（客户端与服务器未建立 Tailscale 直连，
  走 DERP 中继、RTT ~1 s）；服务器侧不是瓶颈。打通直连 / 同网段走 SSH 端口转发可显著提速。
- 微表情 4 列实测 0 非空；手部指标样本极少；这些在报告层按 gap 报"结构性不产出"。

---

## 2026-11-23 全栈升级与挑战杯产品级打磨

### 新模块与基准工具

| 文件 | 作用 |
|---|---|
| `report_frontend/brand.py` + `redline_guard.py` | 统一《衡才智析》品牌与红线词/打分正则硬守卫 |
| `report_frontend/parameter_registry.json` + `parameter_provenance.py` | 21 项关键参数 T1/T2/T3 三级来源登记与常量一致性校验 |
| `report_frontend/cross_modal_coverage.py` | 跨模态逐秒时间窗覆盖率门控（`aligned/degraded/blocked/unverifiable`） |
| `report_frontend/signal_quality_clf.py` + `tools/train_signal_quality_model.py` | 35.3 KB 纯 CPU 自监督采集信号质检分类器（GroupKFold ROC-AUC `0.9996`，严格只读绝不参评） |
| `report_frontend/local_llm_assistant.py` | 隔离按需本地小模型摘要/追问提纲（4s 超时 + 红线硬复检 + 毫秒级确定性回退）与 TF-IDF 题库检索 |
| `tools/jx_cpu_benchmark.py` + `tools/stop_all.sh` | 纯 CPU 端到端性能与峰值内存基准测试 + 一键优雅停机 |

### 新增/加固接口

| 接口 | 说明 |
|---|---|
| `GET /api/system/health` | 并行探活 4 路后端 + FunASR `:10095` 状态与延迟（绕过代理） |
| `POST /api/report/llm-assist` | 按需本地辅助解读（`summary` / `interview_guide`），含合规水印与硬红线复检 |
| `POST /api/question-bank/llm-search` | 毫秒级本地 TF-IDF 题库语义检索（可选本地小模型扩展） |
| `/api/report/structured` · `/api/files/reports` · `/output/*` | 未登录自动脱敏剥离姓名/学号/院系，管理员登录解锁完整信息 |
| `/api/question-bank/*/banks` · `/active` · `:8001/session/<sid>/discard` | 题库写删需管理员鉴权；历史场次丢弃无条件 fail-closed 返回 403 |

### 纯 CPU 轻量化实测（`CUDA_VISIBLE_DEVICES=""`）
面部 `43.39 ms/帧`（`23.05 FPS`） · 手势+全身姿态 `109.66 ms/帧`（`9.12 FPS`） · 语音韵律 `RTF 0.1406`（`7.11×` 实时） · 全链路报告 `176.54 ms` · 同进程峰值内存 `589.5 MB`。

### 新增核心测试
`tests/test_dimension_table_uses_indicator_labels.py`、`tests/test_evidence_chain_enhancements.py`、`tests/test_auth_and_redaction.py`、`tests/test_voice_session_isolation_and_research_series.py`、`tests/test_signal_quality_and_local_assistant.py`。

---

## 2026-11-24 仪器层、特征校准层、不打分 ML 裁判层与多语料科研升级

### 核心升级模块与工具

| 模块 / 工具 | 核心能力与实测口径 |
|---|---|
| `face_expression/` (`4x4` 矩阵真 3D 欧拉角 + 画质门控) | 在 `face_au_log_<sid>.csv` 末尾安全追加 6 列至 98 列契约（`head_yaw_3d`/`pitch`/`roll`、`frame_brightness`、`frame_blur_laplacian`、`frame_quality_ok`）；`tools/jx_backfill_face_3d_telemetry.py` 完成 5 场旗舰历史会话真 3D 回填（`cb79` 切至 `三维姿态矩阵校准`，三维角速度中位数 `0.6352 度/秒`） |
| `voice_interaction/` & `gesture_analysis/` | 语音升级 `Praat-Parselmouth` + 纯 ONNX `Silero VAD` + `CAM++` + `faster-whisper` 词级概率核验；上肢升级 `rtmlib` RTMW 133 点单帧共享引擎（双手稳定检出 `15/15`） |
| `report_frontend/blendshape_calibration.py` + `face_dual_instrument_audit.py` | 质量门控 + Savitzky-Golay 去噪 + P10 基线 + 头姿岭回归残差校准，输出双仪器 5 构念秩一致度与融合判定（`tools/jx_blendshape_calibration_bench.py` 全语料实测 `3d_matrix: 5 / 2d_legacy: 38`） |
| `report_frontend/feature_layer_v2.py` + `behavior_regime_model.py` | 三维角速度、`.npz` 声学动态、`SPARC` 运动平滑、`pycatch22` 22 维动力学（`12.1–20.1 ms`）及 K=4 无监督 GMM 行为节律时间轴 |
| `report_frontend/referee_ml_engine.py` + `tools/train_referee_ml.py` | `GaussianHMM + ruptures.Pelt(rbf)` 阶段切分（5 折抖动降低 `30.11%`，平均驻留 `42.13s`，ARI `0.9732`）；基于下游 G1–G4 真实结局监督（`20 / 12 / 15` 分布，零合成样、彻底去自循环）训练 `IsolationForest + Split Conformal`（`alpha=0.10`，5 折准确率 `0.8936`、Macro-F1 `0.8752`、保形覆盖 `0.9787`、OOD AUROC `0.9802`；7 维纯底层遥测消融准确率 `0.9149`）与白盒可加归因 |
| `tools/jx_gpu_vs_cpu_referee_bench.py` | 分层算力决策验证：常驻单场全套 CPU 仅需 `17.70 ms`（核心推理 `4.85 ms`），Tesla P40 FP32 TCN+BiGRU 预热后 256 场并发达 `2,062.7 场/秒`（较 CPU 提速 `10.48×`） |
| `report_frontend/report_generator.py` & `local_llm_assistant.py` | 离线单文件 HTML 报告内置 5 段「可读行为观察报告」与 3D 校准/`referee_ml` 同源渲染；本地 `:8081 Qwen2.5-14B`（`70s` 超时 + 量纲方向校对 + 红线硬复检 + 确定性规则多段回退）生成导师审阅版 / 个人成长版 / 追问提纲 |
| `report_frontend/scene_profiles.json` & `scene_extensions.py` | 三基地声明式切换：**研思循证 · 科研行为过程分析** / **元智审辨 · AI 元能力与职业实训**（CHI 2024/2025 与 Nelson & Narens 4 项规则指标 + `event_windows` 三段切窗） / **时序观微 · 匿名个体纵向随访**（`anonymous_kinematics_only` 零原片挡位）；保持核心 41 项指标锁死 |
| `report_frontend/longitudinal_tracker.py` & `mcp_career/` | 非遍历性个体（$N$-of-1）纵向追踪引擎：7 项声学与真 3D 运动学指标、EWMA ($\lambda=0.4$ 遇缺口与跨代自动重置)、个人 $\text{Median} \pm 1.4826 \times \text{MAD}$ 稳健偏离指数；4 工具 JSON-RPC 2.0 Career MCP 对齐理论构念且严禁打分 |
| `app.py` (`GET /api/research/benchmarks`) & `research_lab/` | 只读聚合 MIT（`r=0.4023`）、RecruitView（2,011 文本 / 160 多模态）、ChaLearn V2（`1-MAE=0.9126`）与中英文行为节律迁移（联合一致率 `94.86%`）基准，严格不参与实际打分 |

### 本阶段新增测试与门禁实测
- 新增与加强测试：`tests/test_html_report_v2_parity.py`、`tests/test_scene_profiles.py`、`tests/test_mcp_career.py`、`tests/test_longitudinal_tracker.py`、`tests/test_signal_quality_and_local_assistant.py`、`tests/test_backfill_face_3d.py`、`tests/test_referee_ml.py`、`tests/test_blendshape_calibration.py`。
- 全仓验证：全量 `pytest` 实测 `1,759 passed, 1 skipped`；`tools/jx_selfcheck.py --quiet`、`tools/jx_inventory.py --check` 与 `tools/jx_pii_guard.py` 均全绿通过。

# JingXin

一套**结构化行为记录**系统。把一场面试里受试者的面部动作单元、手部/姿态、语音韵律、
话语结构采下来，落成逐帧/逐回答的 CSV，再由报告层按**有出处的判据**披露
「哪些行为线索有证据、哪些没有」。

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
| `recordings/` | 44 场素材（视频/帧/留存账本）。**纯数据** | ❌ 不入库 |
| `.env` | `JX_ADMIN_PASSWORD`。**故意留在仓库外** | ❌ |

仓库内布局：

```
app.py                  Flask 报告面板(:5000)
face_expression/        面部服务(:8000)   core/feature_extraction/au_calculator.py 出 iris 四列+AU
gesture_analysis/       手势服务(:8002)
voice_interaction/      语音服务(:8001)   asr/ 是转写缝(funasr_client → funasr_engine)
report_frontend/        报告层:data_loader → feature_engine → research_mapper → evidence_gate → report_generator
media_retention.py      留存与账本(三服务共用的横切模块)
l0_columns.json         ★ L0 列表的唯一权威(86 列,status × maturity 两轴)
tools/                  起服务 / 验收 / 结构图 / 会话审计 / 转写
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
source tools/jx_env.sh          # 拿到 $JX_PY / $JX_REPO 等

# 起全套(三个采集服务 + 面板 + 前端,幂等)
bash tools/start_all.sh
```

| 服务 | 端口 | 作用 |
|---|---|---|
| face_expression | 8000 | 面部 AU / iris / 头姿 |
| voice_interaction | 8001 | FunASR 转写 + 韵律 |
| gesture_analysis | 8002 | 姿态 + 手部 landmark |
| app.py（面板） | 5000 | 报告面板 API / 文件浏览 |
| 前端（Vite） | 5173 | 界面 |
| FunASR 运行时 | 10095 | 独立部署，见 §5 |

自检：

```bash
curl -s localhost:8000/health; curl -s localhost:8001/health
curl -s localhost:8002/health; curl -s localhost:5000/api/health
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
$PY -m pytest -q                       # 全量
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

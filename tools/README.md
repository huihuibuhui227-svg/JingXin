# tools/ —— 本仓的工具箱

> 2026-09-27 从 `~/shared/`(Windows 侧 `D:\Shared\`)迁入。此前它们**不在 git 里** ——
> 换一台机器就没了,而且是文档里「验收已过」那些话的**证据本体**。

## 先看这个:`jx_env.sh`

所有脚本共用它来解析运行环境。**每样都能用环境变量覆盖**,换机器时基本只需要设这几个:

| 变量 | 默认 | 说明 |
|---|---|---|
| `JX_PY` | `~/miniconda3/envs/jingxin/bin/python` | 本仓解释器 |
| `JX_FRONTEND` | `~/JingXin-frontend` | 前端仓目录 |
| `JX_RECORDINGS` | `~/shared/jingxin_recordings` | 录制根 |
| `JX_FACE_FRAME` | 自动 | 验收用的面部帧 |
| `JX_GESTURE_FRAME` | 自动 | 验收用的手势帧 |
| `JX_AUDIO_SAMPLE` / `JX_AUDIO_SAMPLES` | 自动 | 验收用的音频(单个 / 多段) |
| `JX_CDP` | `http://127.0.0.1:9222` | 浏览器调试端口(录制钩子用) |

**三个样本变量为什么默认自动取**:它们要的是「一张真脸 / 一只手 / 一段真人语音」,
而本仓的原则是**原始媒体不进 git**。所以不给默认文件,而是去 `$JX_RECORDINGS` 下
**最近一场真有该素材的会话**里拿(`media/face/*.jpg`、`media/gesture/*.jpg`、
`media/audio/*_converted.wav`)。

⚠️ 注意是「真有该素材」而不是「目录名最新」—— 验收脚本自己会铸号并建出一个**空壳**
会话目录,它永远比真素材新。2026-09-27 实测踩到过这个坑。

## 起服务

```bash
bash tools/start_all.sh          # 三个服务 + 面板 + 前端,幂等(已在听的端口跳过)
```

- **必须用 `setsid`**(脚本里已经这么做了):否则这些服务留在调用方的进程组里,
  终端一 Ctrl-Z 或被父进程清理时会变 T 态(端口在听却不回话,看着像崩溃)。
  2026-09-27 实测:不带 setsid 时,起完的五个服务在父 shell 退出后**全部消失**。
- 起完用 **Windows 端的浏览器**打开 <http://localhost:5173>。WSL 里的浏览器拿不到摄像头。
- 全停:`for p in 8000 8001 8002 5000 5173; do fuser -k $p/tcp; done`

## 验收(端到端)

按「该先跑哪个」排序 —— 它们都需要服务已经在跑:

| 脚本 | 验什么 | 依赖的样本 |
|---|---|---|
| `tools/m26_media_acceptance.sh` | M2.6 冒烟:原生视频上传端点 + 题目时刻 + 收尾对账 + 路径穿越守卫 | 无(自己造) |
| `tools/m2_acceptance.sh` | M2 契约层:按**前端的调用顺序**模拟一场会话,断言三份日志的文件名与首列都是同一个号 | 面部帧 + 手势帧 + ≥5 段音频 |
| `tools/t7_acceptance.sh` | T7 端到端:M1 + M1.5 的验收门,含**原句不进仓库**的隐私守卫 | 面部帧 + 手势帧 + ≥6 段音频 |
| `tools/m21_acceptance.sh` | M2.1:科研评估自成一场(张冠李戴的守卫) | 1 段音频 + 局域网 FunASR |

```bash
bash tools/m2_acceptance.sh
# 样本不够就显式给:
JX_AUDIO_SAMPLES="a.wav b.wav c.wav d.wav e.wav" bash tools/m2_acceptance.sh
```

⚠️ `m2_acceptance.sh` 有**两个已知卡死点**(记录在案、未修):`start_all.sh` 等 `npm run dev`
的子进程,5173 未先起时走不到断言;第⑤步 `webbrowser.open` 的管道 grep 等不到 EOF。
所以**先单独把 `start_all.sh` 跑完**,再跑它。

## 录制会话(给使用者)

```bash
# ① 一场开始前:铺 meta.json 模板 + 填机器能测到的字段
~/miniconda3/envs/jingxin/bin/python tools/jx_new_session.py <session_id>

# ② 录制期间:把控制台钩子装上(它按原始 CDP 装,刷新也不丢)
~/miniconda3/envs/jingxin/bin/python tools/jx_console_hook.py install
#    想让钩子在 F5 之后依然在,另开一个终端常驻:
~/miniconda3/envs/jingxin/bin/python tools/jx_hook_keeper.py &
#    盯值得看的事件(常驻,给 Monitor 用):
~/miniconda3/envs/jingxin/bin/python tools/jx_console_watch.py

# ③ 录完:核这一场够不够格
~/miniconda3/envs/jingxin/bin/python tools/jx_check_session.py <session_id>
~/miniconda3/envs/jingxin/bin/python tools/jx_check_session.py --all   # 盘上所有会话过一遍
```

## 仓库维护

```bash
~/miniconda3/envs/jingxin/bin/python tools/jx_inventory.py           # 重刷 docs/资源总览.md
~/miniconda3/envs/jingxin/bin/python tools/jx_inventory.py --check   # 只比对,过期退出码 1
~/miniconda3/envs/jingxin/bin/python tools/export_l0_csv.py --out /tmp/l0
```

**`export_l0_csv.py` 为什么是「导出」而不是「存储」**:`~/shared` 原先存着两份手工导出的
`L0列清单_M3.0.csv` / `L0白名单_M3.0.csv`,到第二天就和真源不一致了(85 行 vs 真值 86、
119 条 vs 真值 108)。**同一份数据存两份就会分叉** —— 真源永远是 `l0_columns.json`,CSV 只是视图。

## 其它

- `r5_two_stream_check.html` —— R5(两路 `getUserMedia` 会不会打架)的浏览器自检页。
  要用 **Windows 端浏览器**打开;首次配摄像头时照页面上的六步点一遍。

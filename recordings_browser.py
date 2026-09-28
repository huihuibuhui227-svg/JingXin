# recordings_browser.py
"""素材浏览页的服务端:`/api/recordings*`。

**这个模块做什么**:把录制根(`media_retention.root()`)下的场次列给浏览器,
并在管理员通过密码后,允许看明细、看录像、下日志、翻帧、把不要的目录**移进回收站**。

**三件事决定了它的形状,每一件都不能省**:

1. **目录名里就有身份。** 现在一场叫 `<序号>-<姓名>-<学号>-<院系>__<sid>`
   (2026-09-27 起)。所以"列个目录"这件事本身就是**在列受试者名单**。
   ⟹ 未授权视图在**服务端**就把身份字段剥掉,不是前端藏起来 —— 前端藏等于接口裸奔。
   ⟹ 剥法是**允许清单**(只放行 `_UNAUTH_RECORD_KEYS`),不是禁止清单:新加字段时
     禁止清单会静默失效,允许清单会当场变红。

2. **sid 与文件名都是客户端可控的**,而它们会被拼进路径。
   ⟹ 一律先过 `_resolve_sid()`;`file` 端点再单独过 `_safe_child()` + 后缀允许清单。
   ⟹ 目录**不靠拼** —— 复用 `media_retention.resolve_recording_dir`,不自己再算一遍
     (那四个算过一遍的地方,失效的形态是素材静默写不进去)。

3. **它能删原始素材**,而原始素材没有第二次机会(2026-09-28 丢过 2.1 GB)。
   ⟹ 只有"移进回收站",没有 `rmtree`;且要抄一遍 sid 才执行。
   ⟹ 回收站**嵌一层**:`root()/_回收站/<时间戳>__<原名>/`。嵌一层不是审美 ——
     `resolve_recording_dir` 是在根下按后缀 `__<sid>` 扫的,**平铺会被它扫到**,
     于是同一场以后新录的素材会被写进回收站里(静默,且看着正常)。

⚠️ 本模块**不写**任何素材:唯一会落盘的是 `media/preview/` 下的派生预览件,
   而它绝不碰 `camera.webm` 一个字节(测试比 sha 与 mtime)。
"""

import hmac
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file

import media_retention as mr
import session_purge
from session_purge import NONE_BUCKET, SID_IN_TEXT, SID_PAT

bp = Blueprint("recordings", __name__)

# 管理员密码的环境变量名。**仓库里不写默认值** —— 两个仓都在公开账号下,
# 而写死在前端更糟:那样绕过前端直接打接口就进去了。
ADMIN_PASSWORD_ENV = "JX_ADMIN_PASSWORD"

# sid 形态与 `NONE` 桶名**从 `session_purge` 复用** —— 真删那条路也要认它俩,
# 两处各定义一份就会分叉,而分叉的形态是"一处放行、一处拒绝"。

TRASH_DIR_NAME = "_回收站"
PREVIEW_DIR_NAME = "preview"
PREVIEW_NAME = "camera.preview.webm"
CAMERA_NAME = "camera.webm"
# 告诉调用方这份是**重封装过的**预览件(1)还是退回的原始件(0,拖不动)。
PREVIEW_HEADER = "X-JX-Preview"

TOKEN_TTL_SECONDS = 12 * 3600

# `file` 端点放行的**文件名**与**后缀**。默认拒绝 —— 尤其是 `transcript.json`:
# 里面是受试者的逐字原句,本仓「原句不进仓库」(M1)守的就是它,这一页也不开这个口子。
_FILE_NAME_ALLOWLIST = {"meta.json", "label.json", "session.json"}
_FILE_SUFFIX_ALLOWLIST = {".csv", ".jpg", ".jsonl"}

_UNAUTH_RECORD_KEYS = ("sid", "is_none_bucket", "has_video", "video_bytes",
                       "frames", "bytes_total", "modified", "degraded")

_TOKENS: dict[str, float] = {}
_TOKENS_LOCK = threading.Lock()

# 报告索引:(路径, mtime_ns, size) → [sid, ...]。报告 HTML 的名字是**生成时刻**,
# 不含 sid,所以 sid→报告 只能读内容;按 mtime 缓存,免得每列一次表就全盘读一遍。
_REPORT_INDEX: dict[tuple, list[str]] = {}
_REPORT_INDEX_LOCK = threading.Lock()

_DEGRADED_CACHE: dict[tuple, list[str]] = {}
_DEGRADED_CACHE_LOCK = threading.Lock()


class PreviewUnavailable(RuntimeError):
    """预览件做不出来。**不是错误** —— 退回原始件只是拖不动而已。"""


# ── 鉴权 ──────────────────────────────────────────────────────────────────

def _admin_password() -> str | None:
    return os.getenv(ADMIN_PASSWORD_ENV) or None


def _issue_token() -> str:
    tok = secrets.token_urlsafe(32)
    with _TOKENS_LOCK:
        now = time.time()
        for k, exp in list(_TOKENS.items()):
            if exp <= now:
                _TOKENS.pop(k, None)
        _TOKENS[tok] = now + TOKEN_TTL_SECONDS
    return tok


def _token_is_valid(tok: str | None) -> bool:
    if not tok:
        return False
    with _TOKENS_LOCK:
        exp = _TOKENS.get(tok)
    return exp is not None and exp > time.time()


def _presented_token(*, allow_query: bool = False) -> str | None:
    """凭证的三条来路。头部那两条是给 `fetch`/XHR 用的;query 那条是给**浏览器自己**的。

    ⚠️ 为什么必须有 query 那条:录像与帧是交给浏览器去取的 ——
       `<video src>` / `<img src>` / `<a download>` **都发不了自定义头**。
       绕法只有两个,都不好:
         · 用 axios 取成 blob 再 `createObjectURL` ⟹ 要把整份录像读进内存
           (实测一场 170 MB),而且**拖拽与 Range 全部失效** —— 正好废掉这一页的意义;
         · 让服务端种 cookie ⟹ 要给 CORS 开 `supports_credentials`,牵动四个服务的
           CORS 配置,收益只是"URL 里少一个短时凭证"。
       所以选 query,**并把代价说清楚**:这条路上的 token 会进浏览器历史、
       `<video>` 的请求行、以及面板的访问日志。缓解:它只活 12 小时,且只在 tailnet 内。

    ⚠️ 但它**只开给真正需要的那两个端点**(`allow_query=True` 的那两处):
       `<video>` 与 `<a download>`。JSON 接口一律只认头部 —— 凭证的暴露面越小越好,
       而"浏览器原生取文件"这件事只有那两个端点会遇到。
       收得更紧的做法(一次性票据:`/video?ticket=…`,60 秒有效)接口形状不用改。
    """
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip() or None
    token = request.headers.get("X-Admin-Token")
    if token:
        return token
    return request.args.get("token") if allow_query else None


def _password_missing() -> tuple:
    return jsonify({
        "error": f"这台机器上没有配管理员密码:{ADMIN_PASSWORD_ENV} 未设置。"
                 f"素材浏览的管理员视图因此不可用(不静默放行)。",
    }), 503


def _is_admin(*, allow_query: bool = False) -> bool:
    return _token_is_valid(_presented_token(allow_query=allow_query))


def _admin_required(*, allow_query: bool = False):
    """敏感端点的守卫。**没配密码时同样拒绝**,并且说得清是为什么。

    顺序上先判"密码配没配":没配的话"没有 token"是必然的,报 401 会让人以为
    是自己没登录,而真相是这台机器根本登不进来 —— 两种情形报不同的码,
    否则排查方向会被带偏。
    """
    import functools

    def decorator(view):
        @functools.wraps(view)
        def wrapper(*a, **kw):
            if _admin_password() is None:
                return _password_missing()
            if not _is_admin(allow_query=allow_query):
                return jsonify({"error": "需要管理员密码"}), 401
            return view(*a, **kw)
        return wrapper
    return decorator


# ── 路径守卫 ──────────────────────────────────────────────────────────────

def _resolve_sid(sid: str) -> Path:
    """sid → 场次目录。**先过正则,再用 `media_retention` 的解析,不自己拼路径。**

    解析复用 `resolve_recording_dir`(它认得 `<标签>__<sid>` 与旧的 `root()/<sid>`);
    这里只负责"这个 sid 长不长得像一场"。
    """
    if sid != NONE_BUCKET and not SID_PAT.fullmatch(sid or ""):
        raise ValueError(f"不是合法的 session_id:{sid!r}")
    d = mr.resolve_recording_dir(sid)
    root = mr.root().resolve()
    if not str(d.resolve()).startswith(str(root) + os.sep) and d.resolve() != root:
        # `resolve_recording_dir` 只可能返回根下的路径;真到了这里说明有人改了它。
        raise ValueError(f"解析出的目录在录制根之外:{d}")
    if not d.is_dir():
        raise FileNotFoundError(sid)
    return d


def _safe_child(session_dir: Path, name: str) -> Path:
    """`file` 端点的路径守卫:**解析之后必须仍在场次目录里**,再过文件名允许清单。"""
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise ValueError(f"非法文件名:{name!r}")
    candidate = (session_dir / name).resolve()
    base = session_dir.resolve()
    if candidate != base and not str(candidate).startswith(str(base) + os.sep):
        raise ValueError(f"越出了本场目录:{name!r}")
    if candidate.name == CAMERA_NAME:
        # 原始录像走 /video(要预览与 Range),不走这个通用口子。
        raise ValueError("原始录像请走 /video")
    if candidate.name not in _FILE_NAME_ALLOWLIST and candidate.suffix.lower() not in _FILE_SUFFIX_ALLOWLIST:
        raise ValueError(f"这个文件不在允许清单里:{candidate.name!r}")
    if not candidate.is_file():
        raise FileNotFoundError(name)
    return candidate


# ── 读盘 ──────────────────────────────────────────────────────────────────

def _dir_sid(dir_name: str) -> str | None:
    """目录名 → sid:`<标签>__<sid>` 取后缀;旧的 `root()/<sid>` 取整名。"""
    if dir_name == NONE_BUCKET:
        return NONE_BUCKET
    if SID_PAT.fullmatch(dir_name):
        return dir_name
    if mr.LABEL_DIR_SEP in dir_name:
        tail = dir_name.rsplit(mr.LABEL_DIR_SEP, 1)[-1]
        if SID_PAT.fullmatch(tail):
            return tail
    return None


def _iter_sessions():
    """录制根下的场次目录。跳过回收站(它不是一场,也不该被再删一次)。"""
    root = mr.root()
    if not root.is_dir():
        return
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name == TRASH_DIR_NAME:
            continue
        sid = _dir_sid(p.name)
        if sid is not None:
            yield sid, p


def _label_of(session_dir: Path) -> dict | None:
    f = session_dir / "label.json"
    if not f.is_file():
        return None
    try:
        import json
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _count_frames(session_dir: Path) -> dict:
    out = {}
    for modality in ("face", "gesture"):
        d = session_dir / "media" / modality
        out[modality] = sum(1 for _ in d.glob("*.jpg")) if d.is_dir() else 0
    return out


def _du(path: Path) -> int:
    """整个场次的字节数。**只给明细页用,不许进列表页** —— 它是逐文件 `stat`。

    ⚠️ 实测(2026-09-28):本地 `~/shared` 是 drvfs 挂载(D:\\Shared),那里一次 `stat`
       要 13 ms。列表页对 34 场各做一次 `rglob` + 逐文件 `stat` ⟹ **直接跑不完**。
       服务器是 ext4 会快得多,但那也只是"从跑不完变成每次翻页等几秒",仍然是错的:
       列表的开销必须是 **O(场次数)**,不是 O(全部文件数)。
       由 `test_listing_does_not_walk_every_file` 钉住。
    """
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _degraded_reasons_cached(sid: str, session_dir: Path) -> list[str]:
    """本场的降级原因,按账本的 `(路径, mtime, size)` 缓存。

    为什么值得缓存:列表页**每一场**都要这个数,而读账本是整个文件读进来。
    录完之后账本就不再变 ⟹ 命中率极高。
    ⚠️ 已知缺口:只落在**进程内存**里的降级(`_mark_degraded` 写账失败那一支)
       不进缓存键,所以极端情况下会读到旧值 —— 那一支本身就是"磁盘坏了"。
    """
    try:
        keys = tuple((str(p), (st := p.stat()).st_mtime_ns, st.st_size)
                     for p in sorted(mr.ledger_files(sid)))
    except Exception:
        return []
    cache_key = (sid, keys)
    with _DEGRADED_CACHE_LOCK:
        hit = _DEGRADED_CACHE.get(cache_key)
    if hit is not None:
        return list(hit)
    try:
        reasons = list(mr.degraded_reasons(sid))
    except Exception:
        reasons = []
    with _DEGRADED_CACHE_LOCK:
        if len(_DEGRADED_CACHE) > 512:
            _DEGRADED_CACHE.clear()
        _DEGRADED_CACHE[cache_key] = reasons
    return list(reasons)


def _record(sid: str, session_dir: Path, *, admin: bool,
            report_counts: dict[str, int] | None = None) -> dict:
    """一场的**公开**记录。

    ⚠️ 加字段前先问两件事:
       ① 它是不是身份?是的话只在 `admin=True` 那一支里加 —— 测试用的是允许清单,
          加了会当场变红,这是故意的。
       ② 它要不要遍历文件?列表页每场都要算一次 ⟹ 要遍历的字段(如总体积)
          **只能进明细页**。`_count_frames` 是 readdir,不是逐文件 `stat`,可以在列表里。
    """
    video = session_dir / "media" / CAMERA_NAME
    has_video = video.is_file()
    report_counts = report_counts if report_counts is not None else {}
    reasons = _degraded_reasons_cached(sid, session_dir)
    rec = {
        "sid": sid,
        "is_none_bucket": sid == NONE_BUCKET,
        "has_video": has_video,
        "video_bytes": video.stat().st_size if has_video else 0,
        "frames": _count_frames(session_dir),
        "modified": session_dir.stat().st_mtime,
        "degraded": len(reasons),
        # 有几份报告。**不是身份** ⟹ 未授权也给 —— "哪几场还没出报告"
        # 是排产问题,不是隐私问题。
        "report_count": report_counts.get(sid, 0),
    }
    if admin:
        rec["dir_name"] = session_dir.name
        rec["label"] = _label_of(session_dir)
        rec["degraded_reasons"] = reasons
    return rec


# ── 报告索引 ──────────────────────────────────────────────────────────────

def _output_dir() -> Path:
    """报告落点。**单一来源是 `app.py` 的配置**(它注册本蓝图时写进去的)。"""
    configured = current_app.config.get("OUTPUT_DIR")
    return Path(configured) if configured else (Path(__file__).parent / "data" / "output")


def _logs_dir() -> Path:
    """日志落点(与报告同一讲究:单一来源是 `app.py` 的配置)。"""
    configured = current_app.config.get("LOGS_DIR")
    return Path(configured) if configured else (Path(__file__).parent / "data" / "logs")


def _sids_in_report(f: Path) -> list[str]:
    """一份报告 HTML 里出现的所有 sid(**按 (路径, mtime, size) 缓存**)。

    报告文件名是**生成时刻**、不含 sid,所以 sid→报告 只能读内容;而列一次表要问
    所有报告,不缓存就等于每次翻页把几十份 HTML 全读一遍。
    """
    try:
        st = f.stat()
    except OSError:
        return []
    key = (str(f), st.st_mtime_ns, st.st_size)
    with _REPORT_INDEX_LOCK:
        hit = _REPORT_INDEX.get(key)
    if hit is not None:
        return hit
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    # 扫正文要用**不带锚点**的那个 —— 见 `session_purge.SID_IN_TEXT` 的说明。
    sids = sorted(set(SID_IN_TEXT.findall(text)))
    with _REPORT_INDEX_LOCK:
        _REPORT_INDEX[key] = sids
    return sids


def _all_report_counts() -> dict[str, int]:
    """`sid → 报告份数`。**一个 sid 可能有多份**(同一场跑过几遍)。"""
    d = _output_dir()
    if not d.is_dir():
        return {}
    counts: dict[str, int] = {}
    for f in sorted(d.glob("*Assessment_Report*.html")):
        for sid in _sids_in_report(f):
            counts[sid] = counts.get(sid, 0) + 1
    return counts


def _reports_for(sid: str) -> list[dict]:
    d = _output_dir()
    if not d.is_dir():
        return []
    hits = []
    for f in sorted(d.glob("*Assessment_Report*.html")):
        if sid in _sids_in_report(f):
            try:
                hits.append({"name": f.name, "modified": f.stat().st_mtime})
            except OSError:
                continue
    hits.sort(key=lambda x: x["modified"], reverse=True)
    return hits


# ── 预览件 ────────────────────────────────────────────────────────────────

def _run_ffmpeg_copy(src: Path, dst: Path) -> None:
    """把流式的 MediaRecorder webm 重写一遍,**不重编码**,只补上时长与索引。

    原始件是 live 头(`duration=N/A`、Segment 长度未知、没有 Cues)⟹ 浏览器放得出来
    但拖不动。`-c copy` 秒级完成,内容与原文件一致。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ 临时名**必须留住真正的扩展名**:`.part` 要加在**扩展名之前**。
    #    ffmpeg 是靠**输出文件名的扩展名**猜容器格式的 —— 写成 `x.webm.part` 它会
    #    `Unable to choose an output format` 然后退出 234(实测踩到,而且当时
    #    异常里没带 stderr,所以只看到"预览件没做出来"、看不到为什么)。
    tmp = dst.with_name(f"{dst.stem}.part{dst.suffix}")
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c", "copy", str(tmp)],
            check=True, capture_output=True, timeout=300,
        )
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise PreviewUnavailable("ffmpeg 跑完了但没有产出可用文件")
        os.replace(tmp, dst)      # 原子:半份预览件永远不会被发出去
    except PreviewUnavailable:
        tmp.unlink(missing_ok=True)
        raise
    except subprocess.CalledProcessError as e:
        # ⚠️ **必须把 stderr 带出来**。第一版写的是 `str(e)`,而 `CalledProcessError`
        #    的 str 只有"returned non-zero exit status 234" —— 排查时等于什么都没说。
        #    本仓的规矩是"任何情况都不得静默",这条路径也一样。
        err = (e.stderr or b"").decode("utf-8", "replace").strip()
        tmp.unlink(missing_ok=True)
        raise PreviewUnavailable(
            f"ffmpeg 退出 {e.returncode}:{err[-400:] or '(没有 stderr)'}") from e
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise PreviewUnavailable(f"{type(e).__name__}: {e}") from e


def _preview_is_ready(session_dir: Path) -> bool:
    """预览件已经躺在盘上、而且是照着当前那份原始录像做的。

    比 mtime 而不是"存在就行":录像被换掉之后旧预览件就是**别的一场的画面**,
    而它会照常播出来 —— 那种失效没有任何外部迹象。
    """
    src = session_dir / "media" / CAMERA_NAME
    dst = session_dir / "media" / PREVIEW_DIR_NAME / PREVIEW_NAME
    if not (src.is_file() and dst.is_file()):
        return False
    return dst.stat().st_mtime_ns >= src.stat().st_mtime_ns


def _ensure_preview(session_dir: Path) -> Path | None:
    """返回可发的预览件;做不出来回 `None`(调用方退回原始件)。"""
    src = session_dir / "media" / CAMERA_NAME
    if not src.is_file():
        return None
    dst = session_dir / "media" / PREVIEW_DIR_NAME / PREVIEW_NAME
    if _preview_is_ready(session_dir):
        return dst
    try:
        _run_ffmpeg_copy(src, dst)
    except PreviewUnavailable:
        return None
    return dst


# ── 端点 ──────────────────────────────────────────────────────────────────

@bp.post("/api/admin/login")
def login():
    if _admin_password() is None:
        return _password_missing()
    given = (request.get_json(silent=True) or {}).get("password") or ""
    # 常数时间比较:普通 `==` 会在第一个不同的字节上返回,泄漏前缀。
    if not hmac.compare_digest(given.encode(), (_admin_password() or "").encode()):
        return jsonify({"error": "密码不对"}), 401
    return jsonify({"token": _issue_token(), "expires_in": TOKEN_TTL_SECONDS})


@bp.get("/api/recordings")
def list_recordings():
    admin = _is_admin()
    # **一次**建好 sid→报告数 的映射(缓存过),不要让每一行各扫一遍报告目录。
    report_counts = _all_report_counts()
    recs, total_video, total_frames, trash = [], 0, 0, 0
    for sid, d in _iter_sessions():
        rec = _record(sid, d, admin=admin, report_counts=report_counts)
        total_video += rec["video_bytes"]
        total_frames += sum(rec["frames"].values())
        recs.append(rec)
    trash_dir = mr.root() / TRASH_DIR_NAME
    if trash_dir.is_dir():
        trash = sum(1 for p in trash_dir.iterdir() if p.is_dir())
    recs.sort(key=lambda r: r["modified"], reverse=True)
    # ⚠️ 这里报的是**录像**体积与帧数,不是整个目录的体积 —— 后者要全盘遍历,
    #    进不了列表(见 `_du` 的说明)。整个目录的体积在明细页里给。
    return jsonify({"recordings": recs, "admin": admin,
                    "total_video_bytes": total_video, "total_frames": total_frames,
                    "trash_count": trash})


@bp.get("/api/recordings/<sid>")
@_admin_required()
def recording_detail(sid: str):
    try:
        d = _resolve_sid(sid)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"没有这一场:{sid}"}), 404
    body = _record(sid, d, admin=True)
    # 整个场次的体积只在这一页算 —— 它要逐文件 `stat`,列表页一次都承担不起。
    body["bytes_total"] = _du(d)
    # 预览件备好没有。前端据此决定是"点了就能拖"还是"首次点开会等几秒"——
    # 让用户等的时候知道在等什么,而不是以为页面卡了。
    body["preview_ready"] = _preview_is_ready(d)
    body["files"] = sorted(
        str(p.relative_to(d)) for p in d.rglob("*")
        if p.is_file() and (p.name in _FILE_NAME_ALLOWLIST
                            or p.suffix.lower() in _FILE_SUFFIX_ALLOWLIST)
    )
    body["reports"] = _reports_for(sid)
    return jsonify(body)


@bp.get("/api/recordings/<sid>/video")
@_admin_required(allow_query=True)
def recording_video(sid: str):
    try:
        d = _resolve_sid(sid)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"没有这一场:{sid}"}), 404

    original = d / "media" / CAMERA_NAME
    if not original.is_file():
        return jsonify({"error": "这一场没有原生录像"}), 404

    preview = _ensure_preview(d)
    target = preview or original
    resp = send_file(target, mimetype="video/webm", conditional=True)
    resp.headers[PREVIEW_HEADER] = "1" if preview else "0"
    return resp


@bp.get("/api/recordings/<sid>/frames")
@_admin_required()
def recording_frames(sid: str):
    try:
        d = _resolve_sid(sid)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"没有这一场:{sid}"}), 404

    modality = request.args.get("modality", "face")
    if modality not in ("face", "gesture"):
        return jsonify({"error": f"未知模态:{modality!r}(只允许 face / gesture)"}), 400

    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(120, max(1, int(request.args.get("per_page", 24))))
    except (TypeError, ValueError):
        return jsonify({"error": "page / per_page 得是整数"}), 400

    folder = d / "media" / modality
    names = sorted(p.name for p in folder.glob("*.jpg")) if folder.is_dir() else []
    start = (page - 1) * per_page
    body = {
        "modality": modality, "total": len(names), "page": page, "per_page": per_page,
        "frames": names[start:start + per_page],
    }
    return jsonify(body)


@bp.get("/api/recordings/<sid>/file")
@_admin_required(allow_query=True)
def recording_file(sid: str):
    try:
        d = _resolve_sid(sid)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"没有这一场:{sid}"}), 404
    try:
        target = _safe_child(d, request.args.get("name", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": "没有这个文件"}), 404
    return send_file(target, conditional=True)


@bp.post("/api/recordings/<sid>/trash")
@_admin_required()
def recording_trash(sid: str):
    """移进回收站。**没有 rmtree 这条路** —— 原始素材没有第二次机会。"""
    body = request.get_json(silent=True) or {}
    if (body.get("confirm") or "").strip() != sid:
        return jsonify({
            "error": f"要把 sid 照抄一遍才执行(收到 {body.get('confirm')!r})",
        }), 400

    try:
        d = _resolve_sid(sid)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": f"没有这一场:{sid}"}), 404

    root = mr.root()
    trash_root = root / TRASH_DIR_NAME
    trash_root.mkdir(parents=True, exist_ok=True)
    # 嵌一层(不是平铺)—— 见模块 docstring 第 3 条:`resolve_recording_dir` 会在
    # 根下按后缀扫,平铺的回收站会被它扫到。
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = trash_root / f"{stamp}{mr.LABEL_DIR_SEP}{d.name}"
    n = 1
    while target.exists():
        n += 1
        target = trash_root / f"{stamp}-{n}{mr.LABEL_DIR_SEP}{d.name}"

    os.replace(d, target)          # 同一个盘 ⟹ 原子,不会出现"删了一半"
    return jsonify({"moved_to": str(target.relative_to(root)), "sid": sid})


@bp.post("/api/recordings/<sid>/purge")
@_admin_required()
def recording_purge(sid: str):
    """**真删,没有回收站。** 实现在 `session_purge.purge_session`(与语音服务那条
    入口共用同一份 —— 见那个模块的说明)。

    这一条是给**管理员接口**用的(带密码);录制页在「不留存」那条路上走的是
    语音服务的 `/session/{sid}/discard` —— 因为录制的时候没有管理员 token。
    """
    body = request.get_json(silent=True) or {}
    if (body.get("confirm") or "").strip() != sid:
        return jsonify({"error": f"要把 sid 照抄一遍才执行(收到 {body.get('confirm')!r})"}), 400
    try:
        removed = session_purge.purge_session(sid, logs_dir=current_app.config.get("LOGS_DIR"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except OSError as e:
        return jsonify({"error": f"删除失败:{e}"}), 500
    return jsonify({"purged": sid, **removed})

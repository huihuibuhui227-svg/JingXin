# tests/test_recordings_browser.py
"""素材浏览页(`/api/recordings*`)的守卫。

⚠️ **本文件里的身份数据一律是编的**(`赵六` / `2021004` / `物理学院`),与
   `test_label_dir_naming.py` 用同一套。**不许**从录制盘上抄一个真实目录名当夹具 ——
   第一版就是这么写的(直接用了服务器上一场的目录名),而那几个字是**受试者的
   姓名/学号/院系**。本仓的规矩是「文档/代码里一律不写受试者姓名学号院系,只用
   `session_id`」,测试夹具和文档是同一条规矩。挑夹具时先问:这几个字是从盘上
   抄来的,还是编的?


这一页要做的三件事**每一件都能造成不可逆的后果**,所以每一条守卫都要有钉子:
  1. 它会把**受试者的脸和身份**端到浏览器上 ⟹ 未授权的响应里不许出现任何身份字段;
  2. 它接受客户端传进来的 `sid` 与文件名 ⟹ 必须挡住路径穿越;
  3. 它能**删除原始素材** ⟹ 只能是移到回收站,且不许动录制根以外的任何东西。

⚠️ 本仓最贵的那一类失效是「看着成功、其实没做」,所以下面钉的都是**净行为**:
   不是"某个函数被调用了",而是"盘上到底变成了什么样"。
"""

import hashlib
import subprocess
import json
from pathlib import Path

import pytest

import media_retention as mr
import recordings_browser as rb
import app as panel


_SID = "20260928_212130_5b10"
_LABEL_DIR = f"18-赵六-2021004-物理学院__{_SID}"

# 未授权那次响应里**允许**出现的键。多一个就是泄露 —— 用"允许清单"而不是
# "禁止清单":禁止清单会随着新字段的加入静默失效,允许清单不会。
_UNAUTH_RECORD_KEYS = {
    "sid", "is_none_bucket", "has_video", "video_bytes",
    "frames", "modified", "degraded",
}

# 身份字段的探针。任何一个出现在未授权响应里(哪怕藏在字符串里)都算红。
_IDENTITY_PROBES = ("赵六", "2021004", "物理学院", "student_id", "department",
                    "label", "dir_name")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """一个隔离的录制根。`media_retention.root()` 每次都读环境变量 ⟹ 直接设就行。"""
    r = tmp_path / "recordings"
    r.mkdir()
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(r))
    mr._reset_for_tests()
    return r


def _make_session(root: Path, dir_name: str, *, video=b"", face=0, gesture=0,
                  degraded=(), label=None) -> Path:
    d = root / dir_name
    (d / "media" / "face").mkdir(parents=True)
    (d / "media" / "gesture").mkdir(parents=True)
    for i in range(face):
        (d / "media" / "face" / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff" + bytes([i % 256]))
    for i in range(gesture):
        (d / "media" / "gesture" / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff" + bytes([i % 256]))
    if video:
        (d / "media" / "camera.webm").write_bytes(video)
    if label is not None:
        (d / "label.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    if degraded:
        lines = "".join(json.dumps({"kind": "degraded", "modality": "camera",
                                    "reason": w, "session_id": dir_name.rsplit("__", 1)[-1]},
                                   ensure_ascii=False) + "\n" for w in degraded)
        (d / "retention.camera.jsonl").write_text(lines, encoding="utf-8")
    return d


@pytest.fixture
def client():
    return panel.app.test_client()


@pytest.fixture
def admin_client(root, monkeypatch):
    monkeypatch.setenv(rb.ADMIN_PASSWORD_ENV, "s3cret")
    c = panel.app.test_client()
    tok = c.post("/api/admin/login", json={"password": "s3cret"}).get_json()["token"]
    c.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {tok}"
    return c


# ── 1. 鉴权 ────────────────────────────────────────────────────────────────

def test_login_is_503_when_the_password_is_not_configured(client, monkeypatch):
    """★ 红法:把 `_admin_password()` 的「没设就返回 None」改成给个默认值。

    没设密码时**必须**失败,不能放行 —— 放行的形态是"这台机器上人人都是管理员",
    而它看起来和"配好了"一模一样(页面照常打开、按钮照常能点)。
    """
    monkeypatch.delenv(rb.ADMIN_PASSWORD_ENV, raising=False)
    r = client.post("/api/admin/login", json={"password": "whatever"})

    assert r.status_code == 503, r.get_json()
    assert rb.ADMIN_PASSWORD_ENV in r.get_json()["error"], "得说清是哪个环境变量没设"


def test_login_rejects_a_wrong_password(client, monkeypatch):
    """★ 红法:把 `hmac.compare_digest` 换成 `==`(功能上一样,但那不是红)。"""
    monkeypatch.setenv(rb.ADMIN_PASSWORD_ENV, "s3cret")
    assert client.post("/api/admin/login", json={"password": "nope"}).status_code == 401


def test_a_minted_token_opens_the_admin_view(client, monkeypatch):
    monkeypatch.setenv(rb.ADMIN_PASSWORD_ENV, "s3cret")
    body = client.post("/api/admin/login", json={"password": "s3cret"}).get_json()

    assert body["token"] and len(body["token"]) >= 32, "token 太短/没有 token"
    r = client.get("/api/recordings", headers={"Authorization": f"Bearer {body['token']}"})
    assert r.get_json()["admin"] is True


# ── 2. 未授权视图不许带身份 ────────────────────────────────────────────────

def test_unauthorized_listing_carries_no_identity(client, root):
    """★ 红法:把 `_public_record()` 里剥 label 的那两行去掉(直接返回完整记录)。

    这是本页**唯一一条真正要命的守卫**:目录名里就写着姓名/学号/院系,
    一旦随未授权响应出去,等于把受试者名单挂在了网上。
    """
    _make_session(root, _LABEL_DIR, video=b"v" * 100, face=3, gesture=3,
                  label={"name": "赵六", "student_id": "2021004", "department": "物理学院"})

    raw = client.get("/api/recordings").get_data(as_text=True)
    for probe in _IDENTITY_PROBES:
        assert probe not in raw, f"未授权响应里出现了 {probe!r}:{raw[:300]}"


def test_unauthorized_listing_exposes_only_the_allowlisted_keys(client, root):
    """★ 红法:同上。允许清单比禁止清单硬 —— 新加字段时它会自己变红。"""
    _make_session(root, _LABEL_DIR, video=b"v" * 100, face=3, gesture=3)
    recs = client.get("/api/recordings").get_json()["recordings"]

    assert len(recs) == 1, recs
    assert set(recs[0]) <= _UNAUTH_RECORD_KEYS, set(recs[0]) - _UNAUTH_RECORD_KEYS


def test_unauthorized_still_says_how_much_material_there_is(client, root):
    """未授权 ≠ 什么都看不到:要能回答"一共几场、哪场大、有没有录像"。

    红法:把 `_public_record()` 的 `has_video` / `bytes_total` 也一并剥掉。
    """
    _make_session(root, _LABEL_DIR, video=b"v" * 100, face=3, gesture=3)
    rec = client.get("/api/recordings").get_json()["recordings"][0]

    assert rec["sid"] == _SID, rec
    assert rec["has_video"] is True and rec["video_bytes"] == 100, rec
    assert rec["frames"] == {"face": 3, "gesture": 3}, rec


def test_the_admin_view_does_show_the_label(admin_client, root):
    """红法:两个视图返回同一份数据(未授权被剥、管理员也被剥)。"""
    _make_session(root, _LABEL_DIR, label={"name": "赵六", "student_id": "2021004",
                                           "department": "物理学院"})
    rec = admin_client.get("/api/recordings").get_json()["recordings"][0]

    assert rec["label"]["name"] == "赵六", rec
    assert rec["dir_name"] == _LABEL_DIR, rec


# ── 3. 路径穿越 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    "..", "../..", "../../etc", "..%2f..%2fetc",
    "20260928_212130_5b10/../../..", "/etc/passwd", "a/b",
])
def test_a_malformed_sid_never_reaches_the_filesystem(admin_client, root, bad):
    """★ 红法:撤掉 `media_retention.validate_session_id` 的守卫(或改成直接拼路径)。

    ⚠️ **这一条不是在钉 `_resolve_sid()` 里那个正则** —— 反向复现实测:把那个正则
       改成 `if False`,这一条**照样绿**。因为 `resolve_recording_dir` 自己会先过
       `validate_session_id`,两道守卫叠着,谁也单独失效不了。
       所以这里钉的是**净行为**"没有任何路径逃出录制根";"正则本身有没有被调到"
       由下面那条独立钉住 —— 两条分开,红了才看得出是哪一处。
    """
    for path in (f"/api/recordings/{bad}",
                 f"/api/recordings/{bad}/video",
                 f"/api/recordings/{bad}/frames?modality=face"):
        assert admin_client.get(path).status_code in (400, 404), path

    assert (root / "..").resolve() == root.parent, "守卫不该动到根以外的任何路径"


def test_only_minted_sessions_are_addressable(admin_client, root):
    """★ 红法:把 `_resolve_sid()` 里 `SID_PAT.fullmatch(...)` 那句改成 `if False`。

    反向复现抓出来的:`media_retention.validate_session_id` 只要求
    `[A-Za-z0-9_-]{1,128}`,所以**它挡不住 `my_notes` 这种名字**。少了这一层,
    这一页就退化成"一个能遍历录制根里任意子目录的文件浏览器" ——
    而 `my_notes` 恰好在录制根里躺着、还长得人畜无害。

    `NONE` 是**例外且必须例外**:那是真有素材的桶(没带 session_id 的请求都落那儿),
    列表里也列它,所以它得能被寻址 —— 否则那些素材谁也删不掉。
    """
    (root / "my_notes").mkdir()
    (root / "my_notes" / "meta.json").write_text('{"x":1}', encoding="utf-8")

    assert admin_client.get("/api/recordings/my_notes").status_code == 400
    for path in ("/api/recordings/my_notes/video", "/api/recordings/my_notes/frames",
                 "/api/recordings/my_notes/file?name=meta.json"):
        assert admin_client.get(path).status_code == 400, path

    (root / rb.NONE_BUCKET).mkdir()
    assert admin_client.get(f"/api/recordings/{rb.NONE_BUCKET}").status_code == 200, \
        "NONE 桶里的素材是真素材,不许寻址就等于永远删不掉"


def test_the_file_endpoint_refuses_anything_outside_the_session(admin_client, root, tmp_path):
    """★ 红法:撤掉 `_safe_child()` 里那句"解析之后必须仍在场次目录里"。

    ⚠️ **第一版这条测试没有约束力,反向复现抓出来的**:目标文件当时不存在,
       于是它因为 `FileNotFoundError` 而绿 —— 看起来像守卫生效,其实守卫撤了也绿。
       所以目标必须是**真的存在、且名字刚好在后缀允许清单里**的(`label.json` / `.csv`)。
       这是本仓「fixture 恰好让它过关」那一类,写测试时得盯着"它凭什么绿"。
    """
    _make_session(root, _LABEL_DIR)
    # 同根相邻:在录制根里、但在**本场目录之外**
    (root / "label.json").write_text('{"leak": "同根相邻"}', encoding="utf-8")
    # 根之外
    (tmp_path / "leak.csv").write_text("隔壁,不该被读到", encoding="utf-8")

    for bad in ("../label.json", "media/../../label.json", "../../leak.csv",
                "/etc/passwd", "../../secret.txt"):
        r = admin_client.get(f"/api/recordings/{_SID}/file", query_string={"name": bad})
        assert r.status_code in (400, 404), f"{bad} → {r.status_code}"
        body = r.get_data(as_text=True)
        assert "同根相邻" not in body, f"{bad} 读到了本场目录之外的文件"
        assert "隔壁" not in body, f"{bad} 读到了录制根之外的文件"


def test_the_file_endpoint_serves_an_allowlisted_name(admin_client, root):
    """允许的那几种要真的读得到 —— 否则上面的拒绝测试会因"一律 404"而假绿。"""
    d = _make_session(root, _LABEL_DIR)
    (d / "meta.json").write_text('{"ok": true}', encoding="utf-8")

    r = admin_client.get(f"/api/recordings/{_SID}/file", query_string={"name": "meta.json"})
    assert r.status_code == 200 and r.get_json() == {"ok": True}


# ── 4. 删除:只能移到回收站 ────────────────────────────────────────────────

def test_trash_needs_the_sid_typed_back(admin_client, root):
    """★ 红法:去掉 `confirm` 那一段。

    二次确认的**唯一**意义就是"手滑点不到" —— 少了它,删除按钮和关闭按钮一样好点。
    """
    _make_session(root, _LABEL_DIR)
    r = admin_client.post(f"/api/recordings/{_SID}/trash", json={"confirm": "wrong"})

    assert r.status_code == 400, r.get_json()
    assert (root / _LABEL_DIR).is_dir(), "确认没过就不许动盘"


def test_trash_moves_the_directory_instead_of_deleting_it(admin_client, root):
    """★ 红法:把 `os.replace` 换成 `shutil.rmtree`。

    本仓刚丢过 2.1 GB 且**没有第二次机会**,所以这里要的是"能救回来"。
    """
    d = _make_session(root, _LABEL_DIR, video=b"v" * 10, face=2)
    before = sorted(p.name for p in d.rglob("*"))

    r = admin_client.post(f"/api/recordings/{_SID}/trash", json={"confirm": _SID})
    assert r.status_code == 200, r.get_json()

    assert not (root / _LABEL_DIR).exists(), "原目录还杵在那儿"
    trashed = list((root / rb.TRASH_DIR_NAME).iterdir())
    assert len(trashed) == 1, trashed
    assert trashed[0].name.endswith(_SID), "回收站里的名字要还能认出是哪一场"
    assert sorted(p.name for p in trashed[0].rglob("*")) == before, "内容少东西了"


def test_the_trash_directory_is_not_listed_as_a_session(client, root, admin_client):
    """★ 红法:去掉 `_iter_sessions()` 里跳过回收站那一行。

    否则回收站会以一场"没有录像的会话"的身份出现在列表里 —— 而且它还能被再删一次。
    """
    _make_session(root, _LABEL_DIR)
    (root / rb.TRASH_DIR_NAME / "20260928_2130__x__20260101_000000_aaaa").mkdir(parents=True)

    sids = [r["sid"] for r in client.get("/api/recordings").get_json()["recordings"]]
    assert sids == [_SID], sids


def test_a_trashed_session_is_not_resolved_by_media_retention(admin_client, root):
    """★ 红法:把回收站改成 `root()/<sid>` 下的平铺目录。

    `resolve_recording_dir` 是按**后缀 `__<sid>`** 在根下扫的。回收站嵌一层就避开了它 ——
    这一条钉住那个"嵌一层",因为平铺之后新素材会被写进回收站(静默,且看着正常)。
    """
    _make_session(root, _LABEL_DIR)
    admin_client.post(f"/api/recordings/{_SID}/trash", json={"confirm": _SID})

    resolved = mr.resolve_recording_dir(_SID)
    assert rb.TRASH_DIR_NAME not in resolved.parts, f"回收站里的场次又被解析到了:{resolved}"


# ── 5. 预览件不许碰原始字节 ────────────────────────────────────────────────

def test_making_the_preview_leaves_camera_webm_byte_identical(admin_client, root, monkeypatch):
    """★ 红法:把 ffmpeg 的输出写成 `media/camera.webm`(原地重写)。

    这正是 9-28 事故的形态 —— 而且它回 200、看着完全成功。
    所以这里比的是 sha 与 mtime:**派生件不许动原始素材一个字节**。
    """
    d = _make_session(root, _LABEL_DIR, video=b"original-bytes" * 50)
    original = d / "media" / "camera.webm"
    sha_before = hashlib.sha256(original.read_bytes()).hexdigest()
    mtime_before = original.stat().st_mtime_ns

    # 不去要求这台机器上有 ffmpeg:把外部命令换掉,只验证**路径契约**。
    # 替身也要照契约建父目录 —— 真实那一版就是这么做的(`dst.parent.mkdir`)。
    def _fake(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"preview")

    monkeypatch.setattr(rb, "_run_ffmpeg_copy", _fake)

    r = admin_client.get(f"/api/recordings/{_SID}/video")
    assert r.status_code == 200, r.status_code

    assert hashlib.sha256(original.read_bytes()).hexdigest() == sha_before, "原始录像被动过了"
    assert original.stat().st_mtime_ns == mtime_before, "原始录像的 mtime 变了"
    assert (d / "media" / "preview" / rb.PREVIEW_NAME).is_file(), "预览件没落在预览目录里"


def test_video_falls_back_to_the_original_when_the_preview_cannot_be_made(
        admin_client, root, monkeypatch):
    """★ 红法:`_ensure_preview()` 失败时抛出去(或回 500)。

    ffmpeg 失败不该等于"看不了" —— 原始文件本来就能播,只是拖不动。
    """
    _make_session(root, _LABEL_DIR, video=b"original-bytes" * 50)

    def _boom(src, dst):
        raise rb.PreviewUnavailable("ffmpeg 不在")

    monkeypatch.setattr(rb, "_run_ffmpeg_copy", _boom)
    r = admin_client.get(f"/api/recordings/{_SID}/video")

    assert r.status_code == 200, "预览件做不出来就把整场变成看不了"
    assert r.headers.get(rb.PREVIEW_HEADER) == "0", "要告诉前端这份拖不动"


# ── 5b. 列表页不许全盘遍历 ────────────────────────────────────────────────

def test_listing_does_not_walk_every_file(client, root, monkeypatch):
    """★ 红法:把 `_record` 里那句 `bytes_total: _du(session_dir)` 加回来。

    实测(2026-09-28):本地 `~/shared` 是 drvfs 挂载,`stat` 一次 13 ms;
    列表页对每场 `rglob` + 逐文件 `stat` ⟹ **34 场就跑到超时**。
    服务器是 ext4 会快,但那只是把"跑不完"变成"每次翻页等几秒" —— 仍然是错的。

    所以这里钉的是**开销的量级**:列表页若走了全盘遍历,`rglob` 会被引爆。
    "整个场次多大"这个问题留给明细页(一次只回答一场)。
    """
    for i in range(3):
        _make_session(root, f"2026092{i}_153012_9f3c", video=b"v" * 10, face=4, gesture=4)

    def _explode(self, *a, **kw):
        raise AssertionError(f"列表页遍历了文件:{self}")

    monkeypatch.setattr(Path, "rglob", _explode)
    r = client.get("/api/recordings")

    assert r.status_code == 200, r.get_data(as_text=True)[:200]
    assert len(r.get_json()["recordings"]) == 3


# ── 6. 明细页:降级与报告 ──────────────────────────────────────────────────

def test_the_detail_view_surfaces_the_retention_ledger(admin_client, root):
    """★ 红法:把 `degraded_reasons` 那一行从明细里去掉。

    留存账本此前**没有任何消费方**(§8.1-5 就是抱怨这个)—— 素材到底好不好,
    在这一页必须要能一眼看出来,否则"降级过的一场"与"正常那场"长得一样。
    """
    _make_session(root, _LABEL_DIR, degraded=["camera: 写盘失败", "preflight: 落点不可写"])
    body = admin_client.get(f"/api/recordings/{_SID}").get_json()

    assert len(body["degraded_reasons"]) == 2, body["degraded_reasons"]


# ── 5c. 真 ffmpeg 那一次(替身钉不住的那一类)────────────────────────────

def _ffmpeg_available() -> bool:
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.mark.skipif(not _ffmpeg_available(), reason="这台机器没有 ffmpeg")
def test_a_real_remux_produces_a_seekable_preview(tmp_path):
    """★ 红法:把临时文件名改回 `dst.with_suffix(dst.suffix + ".part")`。

    反向复现抓出来的,而且是**替身测试钉不住的那一类**:前面那条测试把
    `_run_ffmpeg_copy` 换成了一个只会 `write_bytes` 的假函数,于是它永远绿 ——
    而真实调用在**第一次上真素材时就失败了**:
        ffmpeg 靠**输出文件名的扩展名**猜容器格式,`x.webm.part` 它不认识,
        于是 `Unable to choose an output format`、退出 234。
    教训:凡是"我们只是转交给外部命令"的那一步,替身测的是**我们的调用形状**,
    不是那个命令能不能跑 —— 所以至少要有一次**不带替身**的。

    这里顺带钉住"做出来的确实是可拖拽的那一份"(有时长),因为
    "文件存在"和"能拖"是两件事,而这一页做预览的全部意义就是后者。
    """
    src = tmp_path / "in.webm"
    made = subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=5",
         "-c:v", "libvpx", str(src)],
        capture_output=True,
    )
    if made.returncode != 0:
        pytest.skip(f"这台机器做不出测试素材:{made.stderr.decode()[:120]}")

    # 原始件的特征:MediaRecorder/流式头 ⟹ ffprobe 说不出时长
    def _duration(p) -> str:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(p)], capture_output=True, text=True)
        return r.stdout.strip()

    dst = tmp_path / "out" / rb.PREVIEW_NAME
    rb._run_ffmpeg_copy(src, dst)

    assert dst.is_file() and dst.stat().st_size > 0, "预览件没产出"
    assert _duration(dst) not in ("", "N/A"), f"预览件说不时长,也就拖不动:{_duration(dst)}"
    assert not list(dst.parent.glob("*.part*")), "临时文件没清干净"


@pytest.mark.skipif(not _ffmpeg_available(), reason="这台机器没有 ffmpeg")
def test_a_failed_remux_says_why(tmp_path):
    """★ 红法:把 `CalledProcessError` 那一支换回 `raise PreviewUnavailable(str(e))`。

    第一版就是这么写的,于是真素材上第一次失败时,能看到的只有
    "returned non-zero exit status 234" —— 排查等于从零开始。
    """
    src = tmp_path / "not-a-video.webm"
    src.write_bytes(b"this is definitely not a video")

    with pytest.raises(rb.PreviewUnavailable) as excinfo:
        rb._run_ffmpeg_copy(src, tmp_path / "out" / rb.PREVIEW_NAME)

    msg = str(excinfo.value)
    # ⚠️ 断言必须落在 **ffmpeg 自己说的话** 上,不能用 "error" 这种裸词 ——
    #    命令行里本来就有 `-v error`。反向复现实测:第一版用裸词断言时,
    #    "把 stderr 整段吞掉、只留 `str(CalledProcessError)`" 的变异体**照样绿**。
    #    (这正是本仓记过的那个形态:断言用了裸词,而别处本来就有那个词。)
    assert "Invalid data found" in msg, f"没带上 ffmpeg 的原话:{msg}"
    assert "183" in msg, f"没带上退出码:{msg}"


def test_preview_ready_goes_stale_when_the_video_is_replaced(admin_client, root):
    """★ 红法:把 `_preview_is_ready` 换成"文件存在就算数"。

    录像被换掉之后,旧预览件就是**别的一场的画面** —— 而它照常播得出来,
    没有任何外部迹象。所以判据必须是"照着**当前**这份录像做的"(比 mtime)。
    """
    import os
    d = _make_session(root, _LABEL_DIR, video=b"v" * 50)
    assert admin_client.get(f"/api/recordings/{_SID}").get_json()["preview_ready"] is False

    pv = d / "media" / rb.PREVIEW_DIR_NAME
    pv.mkdir(parents=True)
    (pv / rb.PREVIEW_NAME).write_bytes(b"preview")
    assert admin_client.get(f"/api/recordings/{_SID}").get_json()["preview_ready"] is True

    # 原始录像被换掉(新的更新)⟹ 旧预览件立即失效
    src = d / "media" / rb.CAMERA_NAME
    src.write_bytes(b"v" * 80)
    os.utime(src, (src.stat().st_atime + 10, src.stat().st_mtime + 10))
    assert admin_client.get(f"/api/recordings/{_SID}").get_json()["preview_ready"] is False


def test_every_sensitive_endpoint_refuses_without_a_token(client, root, monkeypatch):
    """★ 红法:给任何一个端点去掉 `@_admin_required`。

    密码**配好了**、只是没带 token ⟹ 401(说的正是"你没登录")。
    没配密码那一种走下面那条 —— 两者报的码不同,是因为要传达的事不同。
    """
    monkeypatch.setenv(rb.ADMIN_PASSWORD_ENV, "s3cret")
    _make_session(root, _LABEL_DIR, video=b"v" * 10, face=1)
    for path in (f"/api/recordings/{_SID}", f"/api/recordings/{_SID}/video",
                 f"/api/recordings/{_SID}/frames?modality=face",
                 f"/api/recordings/{_SID}/file?name=label.json"):
        assert client.get(path).status_code == 401, path
    assert client.post(f"/api/recordings/{_SID}/trash", json={"confirm": _SID}).status_code == 401


def test_media_endpoints_also_accept_the_token_in_the_query(client, root, monkeypatch):
    """★ 红法:把 `_presented_token()` 里 `request.args.get("token")` 那一支去掉。

    `<video src>` / `<img src>` / `<a download>` 都发不了自定义头 —— 这是**浏览器**
    的限制,不是选择。少了这一支,录像在页面上根本播不出来。

    ⚠️ 代价要记住:这条路上的 token 会进浏览器历史与服务端访问日志。
       所以下一个断言钉住**暴露面** —— 只有这两个端点吃 query,别的都只认头部。
    """
    monkeypatch.setenv(rb.ADMIN_PASSWORD_ENV, "s3cret")
    d = _make_session(root, _LABEL_DIR, video=b"v" * 10, face=1)
    (d / "meta.json").write_text("{}", encoding="utf-8")
    tok = client.post("/api/admin/login", json={"password": "s3cret"}).get_json()["token"]

    assert client.get(f"/api/recordings/{_SID}/video?token={tok}").status_code == 200
    assert client.get(f"/api/recordings/{_SID}/file?name=meta.json&token={tok}").status_code == 200

    # 其余端点即使带了正确的 token 也不认 query —— 凭证的暴露面越小越好。
    # 列表端点本身是**公开**的(未授权也要能看"有几场"),所以它回 200 而非 401;
    # 要钉的是它**忽略**了那个 token:`admin` 必须仍是 false。
    assert client.get(f"/api/recordings?token={tok}").get_json()["admin"] is False, \
        "列表端点认了 query token —— 那等于把凭证塞进了最常见的那个 URL"
    for path in (f"/api/recordings/{_SID}?token={tok}",
                 f"/api/recordings/{_SID}/frames?modality=face&token={tok}"):
        assert client.get(path).status_code == 401, path


def test_sensitive_endpoints_say_503_when_no_password_is_configured(client, root, monkeypatch):
    """★ 红法:把 `_admin_required` 里判"密码配没配"那一段去掉。

    没配密码时报 401 会让人以为是自己没登录,而真相是**这台机器根本登不进来** ——
    两种情形要报不同的码,否则排查方向会被带偏。
    """
    monkeypatch.delenv(rb.ADMIN_PASSWORD_ENV, raising=False)
    _make_session(root, _LABEL_DIR, video=b"v" * 10, face=1)
    for path in (f"/api/recordings/{_SID}", f"/api/recordings/{_SID}/video"):
        r = client.get(path)
        assert r.status_code == 503, path
        assert rb.ADMIN_PASSWORD_ENV in r.get_json()["error"]

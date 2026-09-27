from typing import Optional, Dict
from ...models.features import AUFeatures, TemporalStats, MicroExpressionResult
from ...models.results import EmotionResult

# ══════════════════════════════════════════════════════════════════════════════
# ★★ 过期门限账本(2026-09-27)——**这 25 条比较里的绝大多数,是照旧口径的数字范围定的**
# ══════════════════════════════════════════════════════════════════════════════
#
# 背景:本模块 `infer()` 里的门限(`au26 > 0.3`、`au6 > 0.1`、`au12 > 0.1` …)写下来的时候,
# 那些 AU 列的口径**不是现在的口径**。之后两批改动把它们换掉了:
#   · B2/B3(2026-09-27):`au6_cheek_raise`(分母换两次,末次 = 面部轮廓高)、
#     `au25_mouth_open`(唇内缘原始比值、不减基线、不截顶)、`au26_jaw_drop`(分母 `0.1×`→
#     轮廓高、不减基线、不截顶)、`au4_frown`(改比值本身、÷ 眼距)、
#     `au9_nose_wrinkle`(换鼻翼下标)、`au23_lip_compression`(改唇红厚度 ÷ 尺度)、
#     `symmetry_score`(开始除尺度);
#   · 本批(2026-09-27):面部尺度的分母端点由**鼻尖**换成**前额顶**(面部轮廓高),
#     于是以它为分母的每一列(au1/au2/au10/au15/au23/au25/au26、head_pitch、
#     symmetry_score)一起换尺度,`au12_smile` 的基线也由「开头 10 帧冻结均值」改成
#     **全程口宽 p10**(并去掉截顶)。
#
# 后果(三场正式素材重放;有脸帧 271/477/479;`实验:` `experiments/replay_retained.py`,
# 喂留存账本的 `declared_ts`)——**分支命中帧数当场变了**,而且有两处不只是"变了",
# 是**结构性死掉/恒真**:
#   · `au23 > 0.2`:三场命中 **0/0/0**(现口径下 au23 值域约 0.06~0.20,顶格 0.198)
#     ⟹ `forced_smile` 分支**永远不可能命中**(改前 16/287/165);`anxiety` 的 au23 项也成了死项。
#   · `au6 > 0.1`:三场 **271/477/479 = 全部命中**(现口径 au6 ≈ 0.73~0.81,最小值 0.728)
#     ⟹ 这一项在 `happy` 里**永不可能是决定性的那一项**,并且把 `polite_smile`
#     (它的 `elif` 要求 `au6 <= 0.1`)变成**死分支**(改前 au6 值域含 0.03~0.68,曾命中 7 帧)。
#   · `au26 > 0.3`:三场 **全部命中**(现口径 au26 ∈ [0.40, 0.72],改前 ∈ [0, 1])。
#   · `au4 > 0.1` / `> 0.2`:改前命中 7/2/91 与 2/0/9,现在 **全部命中**
#     ⟹ `anger` 从"几乎从不命中"(改前 emotion_anger = 0/0/0)变成几乎全命中(267/471/437)。
#
# **口径(使用者已裁定,2026-09-27)**:这些门限**不重新标定** ——
# 重登记需要一个**标定集**(本仓没有;M3.1 已被裁定跳过),而"编一个阈值"正是
# `l0_columns.json` 反复禁止的形态。按 spec,门限重登记归 **M3.5**。
# 所以本账本的作用不是"修好门限",而是**让「它们已过期」不可能被静默忽略**:
#   ① 每一条门限都登记在这里 —— 它照哪个旧口径写的、该列现在的口径、实测后果;
#   ② `tests/test_emotion_threshold_staleness.py` 用 **AST** 从 `infer()` 的源码里推出全部
#      「变量 vs 浮点常量」的比较,与本账本**双向**比对:门限改了/加了/删了而账本没动 ⟹ 红;
#   ③ 同一文件里两条边界用例**走真产出方**(`VideoPipeline` → 本模块),任何对
#      au12/au26 口径的后续改动都会翻转分支 ⟹ 红 ⟹ 下一个人必须来重登记。
#
# ⚠️ 两条如实说明(别把本账本读大):
#   · `au25_mouth_open` 的口径也变了,但它在本模块里**只是一个没有被读过的赋值**
#     (`au25 = au_features.au25_mouth_open` 之后再没出现)⟹ 它的口径变化**对本模块零影响**。
#     这里如实记下,免得下一个人以为"au25 也过期了"而去改一条死代码。
#   · `emotion_*` 分量在报告层**永久封停**(`report_frontend/evidence_gate.QUARANTINE` 的
#     `emotion_` 键,`permanent=True`)⟹ 这些错配**进不了打分**。但它们**进得了**
#     `tension_score`(`tension_engine` 读 emotion_vector 的 anxiety/anger/fear/moral_disgust),
#     而 `tension_score` 只是**暂时**封停(unblock = M3)⟹ 解封那天这条错配就会浮上来。
STALE_THRESHOLDS: dict[str, dict] = {
    # ⚠️ 所有 `measured` 都是**三场正式素材**(20260926_153202_2b11 / 153854_1592 /
    #    155559_caf0;有脸帧 271/477/479)重放出来的**分支命中帧数或门限满足帧数**,
    #    重放腿 = `experiments/replay_retained.py`,喂留存账本的 `declared_ts`。
    #    三个时间点对比:"旧" = au6/au25/au26/au4/au9/au23/symmetry 换口径**之前**
    #    (盘上 2026-09-26 那次重放),"本批改前" = B2/B3 已落、面部尺度换端点**之前**,
    #    "本批改后" = 现在。
    "au12 > 0.1": {
        "column": "au12_smile",
        "written_against": "旧:基线 = 开头 10 帧冻结均值、比值截顶到 1(值域 [0,1])",
        "now": "基线 = **全程口宽 p10**(因果近似)、不截顶(值域可 > 1)",
        "measured": "三场满足帧数 90/414/251(本批改前) → **151/344/322**(本批改后);"
                    "`emotion_happy` 90/414/252 → 152/349/323。⟹ 门限本身没动,但比值整体变大,"
                    "命中面显著扩大",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au6 > 0.1": {
        "column": "au6_cheek_raise",
        "written_against": "旧:÷ face_height(鼻尖→下巴),值域 ~0.03~0.68",
        "now": "÷ **面部轮廓高** dist(lm[10], lm[152]);三场 ∈ [0.728, 0.813]",
        "measured": "三场满足帧数 271/470/479(旧) → 271/477/479(**全部**,本批改前/改后同)"
                    "⟹ **恒真**;并把 `polite_smile`(它的 elif 要求 au6 <= 0.1)变成**死分支**"
                    "(改前 `emotion_polite_smile` 曾命中 0/7/0,现在 0/0/0)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au26 > 0.3": {
        "column": "au26_jaw_drop",
        "written_against": "旧:减开头 10 帧基线、÷ (0.1×face_height)、截顶到 1(值域 [0,1])",
        "now": "|下巴−上唇内缘| ÷ **面部轮廓高**,不减基线、不截顶;三场 ∈ [0.110, 0.417]",
        "measured": "三场满足帧数 189/308/131(旧) → 271/477/479(全过,**恒真**) → "
                    "**87/24/182**(本批改后)⟹ 同一个 0.3 先在换口径时变成恒真、又在换尺度后"
                    "变成「偶尔过」;`emotion_surprise` 189/269/121 → 271/413/416 → **9/2/20**",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au26 > 0.5": {
        "column": "au26_jaw_drop",
        "written_against": "同 `au26 > 0.3`(旧口径下的 0.5 是「半程」位置)",
        "now": "同 `au26 > 0.3`(三场 max = 0.356/0.356/0.417 ⟹ **顶不到 0.5**)",
        "measured": "三场满足帧数 163/224/91(旧) → 266/469/461(本批改前) → **0/0/0**"
                    "(本批改后)⟹ **结构性不可达**:`emotion_fear` 203/132/219 → **0/0/0** "
                    "(分支死掉,与 `au23 > 0.2` 同一形态)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au23 > 0.2": {
        "column": "au23_lip_compression",
        "written_against": "旧:`1 − 当前唇厚/静息唇厚`(值域 [0,1],且与 au25 互为镜像)",
        "now": "唇红厚度 dist(lm[0], lm[13]) ÷ **面部轮廓高**;三场 ∈ [0.016, 0.085]",
        "measured": "三场满足帧数 65/330/316(旧) → 0/0/0(本批改前) → **0/0/0**(本批改后)"
                    "⟹ **结构性不可达**;`emotion_forced_smile` 16/287/165 → **0/0/0**"
                    "(分支死掉),`anxiety` 的 au23 项也成了死项",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au4 > 0.3": {
        "column": "au4_frown",
        "written_against": "旧:`1 − dist(lm[276], lm[33]) ÷ face_width`(实测 86~92% 触地板)",
        "now": "dist(lm[107], lm[336]) ÷ 眼距(**比值本身**);三场 ∈ [0.250, 0.369]",
        "measured": "三场满足帧数 0/0/0(旧) → 267/471/441(本批改前/改后同)⟹ `emotion_anger`"
                    "从**从未命中**(0/0/0)变成几乎全命中(267/471/437);"
                    "`moral_disgust` / `cognitive_load` 的 au4 项同样翻面",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au4 > 0.2": {
        "column": "au4_frown",
        "written_against": "同 `au4 > 0.3`",
        "now": "同 `au4 > 0.3`(三场 min = 0.250 ⟹ **恒真**)",
        "measured": "三场满足帧数 2/0/9(旧) → 271/477/479(**全部**)⟹ 恒真"
                    "(`emotion_sadness` 的 au4 项因此不再筛任何帧 —— 那条分支只剩 au15 一个门)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au4 > 0.1": {
        "column": "au4_frown",
        "written_against": "同 `au4 > 0.3`",
        "now": "同 `au4 > 0.3`",
        "measured": "三场满足帧数 7/2/91(旧) → 271/477/479(**全部**)⟹ 恒真"
                    "(`anxiety` 的 au4 项)⟹ `emotion_anxiety` 从 73/336/363 变成字面意义上的"
                    "**全帧命中**(271/477/479)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au9 > 0.2": {
        "column": "au9_nose_wrinkle",
        "written_against": "旧:鼻翼读 lm[234]/lm[455](每视频 max 恰为 0.5)",
        "now": "鼻翼读 NOSE_WING_LEFT/RIGHT(129/358);三场 ∈ [0, 0.854]",
        "measured": "三场满足帧数 271/475/460(旧) → 271/476/466(本批改前/改后同)⟹ 值域整体"
                    "上移(max 0.5 → 0.85),这条在现口径下**几乎恒真**",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au9 > 0.3": {
        "column": "au9_nose_wrinkle",
        "written_against": "同 `au9 > 0.2`(旧口径 max = 0.5 ⟹ 0.3 落在上四分位)",
        "now": "同 `au9 > 0.2`",
        "measured": "三场满足帧数 271/471/457(旧) → 271/474/458(本批改前/改后同)⟹ 同样几乎恒真",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au10 > 0.1": {
        "column": "au10_upper_lip_raise",
        "written_against": "旧:÷ face_height(鼻尖→下巴)",
        "now": "÷ **面部轮廓高**(分母变大 ⟹ 同一个分子给出的比值整体变小)",
        "measured": "三场满足帧数 183/342/132(本批改前) → **140/316/94**(本批改后);"
                    "`emotion_disgust` 的 au9 项几乎恒真,所以那条分支基本只由这一项决定",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au10 > 0.2": {
        "column": "au10_upper_lip_raise",
        "written_against": "同 `au10 > 0.1`",
        "now": "同 `au10 > 0.1`",
        "measured": "三场满足帧数 144/316/95(本批改前) → **71/308/49**(本批改后)"
                    "(`moral_disgust` 的 au10 项)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au15 > 0.05": {
        "column": "au15_mouth_down",
        "written_against": "旧:÷ face_height 且带静息基线(本列是 `blocked` 行,只吐原始量)",
        "now": "÷ **面部轮廓高**;静息基线机制未变",
        "measured": "三场满足帧数 59/124/285(本批改前) → **47/109/266**(本批改后)"
                    "(`emotion_sadness` / `distress` 的门)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au20 > 0.1": {
        "column": "au20_lip_stretcher",
        "written_against": "旧:与 au12 同一行代码 —— 基线 = 开头 10 帧冻结均值、截顶到 1",
        "now": "与 au12 一起改成**全程口宽 p10**、不截顶(`legacy_allowlist` 已登记 M3.2 删列)",
        "measured": "三场满足帧数 90/414/251(本批改前) → **151/344/322**(本批改后);"
                    "⚠️ 它所在的那条分支(`forced_smile`)已被 `au23 > 0.2` 卡死(见上)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au1 > 0.1": {
        "column": "au1_inner_brow_raise",
        "written_against": "旧:÷ face_height(鼻尖→下巴),三场均值 0.170/0.151/0.154",
        "now": "÷ **面部轮廓高**,三场均值 0.080/0.069/0.076(值整体减半)",
        "measured": "三场满足帧数 271/413/415(本批改前) → **6/48/53**(本批改后)⟹ 从恒真"
                    "变成几乎恒假;`emotion_surprise` 因此 271/413/416 → 9/2/20",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au1 > 0.15": {
        "column": "au1_inner_brow_raise",
        "written_against": "同 `au1 > 0.1`",
        "now": "同 `au1 > 0.1`(三场 max = 0.150/0.132/0.148 ⟹ **顶不到 0.15**)",
        "measured": "三场满足帧数 210/184/236(本批改前) → **0/0/0**(本批改后)⟹ "
                    "**结构性不可达**;`emotion_fear` 因此与 `au26 > 0.5` 一起变成**双死门**"
                    "(203/132/219 → 0/0/0)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au2 > 0.1": {
        "column": "au2_outer_brow_raise",
        "written_against": "同 `au1 > 0.1`(本列在 `legacy_allowlist` 里,planned = M3.2 删列)",
        "now": "同 `au1 > 0.1`",
        "measured": "三场满足帧数 249/221/265(本批改前) → **26/86/59**(本批改后)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "au2 > 0.15": {
        "column": "au2_outer_brow_raise",
        "written_against": "同 `au2 > 0.1`",
        "now": "同 `au2 > 0.1`",
        "measured": "三场满足帧数 155/146/147(本批改前) → **1/4/9**(本批改后)",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "symmetry < 0.7": {
        "column": "symmetry_score",
        "written_against": "旧:`1 − (两个竖直差之和)`,**不除尺度**(实测 mean 0.9825,近常量)",
        "now": "`1 − (…) ÷ 面部轮廓高`(分母换端点后值整体上移:三场 mean 0.955/0.907/0.927)",
        "measured": "三场满足帧数 0/0/0(旧) → 1/80/48(本批改前) → **1/0/0**(本批改后)⟹ "
                    "换尺度后这条又几乎恒假;`emotion_contempt` 1/69/40 → 1/0/0",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
    "indirect": {
        "column": "au1_inner_brow_raise / au4_frown",
        "written_against": "`au1_inner_brow_raise_trend > 0.01` 与 `au4_frown_trend > 0.01`"
                           "(trend = 窗内斜率 ×100,见 `video_pipeline` 的时序统计)",
        "now": "au1 的尺度本批变了(分母换端点,值整体减半)⟹ 同样一次面部动作给出的 trend 变小;"
               "au4 的尺度在 B2 变了(值域 0.03~0.37 → 0.25~0.37)",
        "measured": "`emotion_startled_anxiety` 84/133/160(本批改前) → **85/126/147**(本批改后)"
                    "—— 只挪了一点,但这条判据两端都是**照旧尺度定的 0.01**。"
                    "⚠️ **本条不在 AST 判据的覆盖范围内**(那两条比较的左边是**调用**、不是变量),"
                    "单列在此,免得「AST 抓不到」被读成「它们没问题」",
        "status": "门限待 M3.5 重登记(不许自行编新阈值:本仓没有标定集)",
    },
}

# 不受本批/B2 口径影响的门限 —— 也登记(否则 AST 那条双向钉子会把它们报成"漏登记")。
UNAFFECTED_THRESHOLDS: dict[str, str] = {
    "au7 > 0.3": "au7 = 1 − avg_ear,EAR 的算法与下标本批/B2 都没动",
    "avg_ear < 0.15": "同上(EAR 公式未动);三场命中 0/0/2,`fatigue` 本来就极少命中",
    "head_yaw > 0.1": "head_yaw 除的是 face_width(本批未动);⚠️ 但整列被判 `blocked`"
                      "(2D 代理 → 真 3D),换掉它时这三条也要重登记",
    "head_yaw < -0.1": "同 `head_yaw > 0.1`",
    "head_yaw < -0.05": "同 `head_yaw > 0.1`",
    "v > 0.3": "`composite_emotions` 的筛选:比的是**归一化后**的分量占比(0~1),"
               "与任何 AU 的口径无关",
}


class EmotionEngine:
    def infer(
        self,
        au_features: AUFeatures,
        temporal_stats: Optional[TemporalStats] = None,
        micro_expressions: Optional[MicroExpressionResult] = None
    ) -> EmotionResult:
        emotions = {
            "happy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0,
            "disgust": 0.0, "contempt": 0.0, "anxiety": 0.0, "fatigue": 0.0,
            "polite_smile": 0.0, "distress": 0.0,
            "forced_smile": 0.0,
            "startled_anxiety": 0.0,
            "cognitive_load": 0.0,
            "moral_disgust": 0.0
        }

        au1 = au_features.au1_inner_brow_raise
        au2 = au_features.au2_outer_brow_raise
        au4 = au_features.au4_frown
        au6 = au_features.au6_cheek_raise
        au7 = au_features.au7_eye_squeeze
        au9 = au_features.au9_nose_wrinkle
        au10 = au_features.au10_upper_lip_raise
        au12 = au_features.au12_smile
        au14 = au_features.au14_dimpler
        au15 = au_features.au15_mouth_down
        au20 = au_features.au20_lip_stretcher
        au23 = au_features.au23_lip_compression
        au25 = au_features.au25_mouth_open
        au26 = au_features.au26_jaw_drop
        symmetry = au_features.symmetry_score
        head_yaw = au_features.head_yaw
        avg_ear = au_features.avg_ear

        # ★★ 下面这些常量**照旧口径的数字范围**定的,而 au12/au4/au6/au23/au26/symmetry/
        # au1/au2/au10/au15/au20 的口径在 B2/B3 与本批里都改过 —— 逐条的账
        # (照哪个旧口径写的、现在为什么对不上、实测后果)**在模块顶部的
        # `STALE_THRESHOLDS` 里**,不在每一行重复。⚠️ 门限**不重新标定**:
        # 那要标定集(本仓没有),按 spec 归 **M3.5**。改动本段任何一条比较 ⟹
        # `tests/test_emotion_threshold_staleness.py` 会红(它用 AST 从本函数的源码里
        # 推出全部门限与账本双向比对)。

        # ⚠️ `au6 > 0.1`:现口径下 au6 ≈ 0.73~0.81 ⟹ **恒真**(三场 271/477/479 全部命中);
        #    它把 `polite_smile`(下面的 elif)变成**死分支**。账见 `STALE_THRESHOLDS`。
        if au12 > 0.1 and au6 > 0.1:
            emotions["happy"] = min((au12 + au6) / 2, 1.0)
        elif au12 > 0.1:
            emotions["polite_smile"] = au12 * 0.7

        # ⚠️ `au26 > 0.3`:现口径(÷ 面部轮廓高、不减基线、不截顶)下 au26 ∈ [0.40, 0.72]
        #    ⟹ 三场全部命中、**恒真**(改前它筛掉 189/308/131 帧里的另一半)。账见账本。
        if (au1 > 0.1 or au2 > 0.1) and au26 > 0.3:
            emotions["surprise"] = min((au1 + au2 + au26) / 3, 1.0)

        if au9 > 0.2 and au10 > 0.1:
            emotions["disgust"] = min((au9 + au10) / 2, 1.0)

        if au4 > 0.3 and au7 > 0.3:
            emotions["anger"] = (au4 + au7) / 2

        if au4 > 0.2 and au15 > 0.05:
            sadness = (au4 + au15) / 2
            emotions["sadness"] = sadness
            emotions["distress"] = sadness * 0.8

        if (au1 > 0.15 or au2 > 0.15) and au26 > 0.5 and head_yaw > 0.1:
            # ⚠️ `au26 > 0.5`:现口径下三场 min = 0.403/0.442 ⟹ 这条几乎恒真;而它
            #    "筛掉一半"的那个行为是**旧口径**下的(命中 163/224/91)。账见账本。
            emotions["fear"] = min((au1 + au2 + au26) / 3, 1.0)

        if avg_ear < 0.15:
            emotions["fatigue"] = 1.0 - avg_ear

        if au12 > 0.1 and au20 > 0.1 and au23 > 0.2:
            # ⚠️ `au23 > 0.2`:现口径(唇红厚度 ÷ 面部轮廓高,三场 ≤ 0.198)下
            #    **结构性不可达** ⟹ 这条分支**永远不命中**(改前 16/287/165)。账见账本。
            emotions["forced_smile"] = min((au12 + au20 + au23) / 3, 1.0)

        if temporal_stats and \
           temporal_stats.data.get('au1_inner_brow_raise_trend', 0) > 0.01 and \
           temporal_stats.data.get('au4_frown_trend', 0) > 0.01:
            emotions["startled_anxiety"] = 0.7

        if au4 > 0.2 and au7 > 0.3 and head_yaw < -0.1:
            emotions["cognitive_load"] = min((au4 + au7) / 2, 1.0)

        if au9 > 0.3 and au10 > 0.2 and au4 > 0.3:
            emotions["moral_disgust"] = min((au9 + au10 + au4) / 3, 1.0)

        if au12 > 0.1 and symmetry < 0.7:
            emotions["contempt"] = min(au12 * (1 - symmetry), 1.0)

        anxiety_score = 0.0
        if au4 > 0.1:
            anxiety_score += au4 * 0.5
        if au23 > 0.2:
            anxiety_score += (au23 - 0.2) * 1.5
        if head_yaw < -0.05:
            anxiety_score += 0.2
        emotions["anxiety"] = min(anxiety_score, 1.0)

        total = sum(emotions.values())
        if total > 0:
            for k in emotions:
                emotions[k] = round(emotions[k] / total, 3)
        else:
            emotions["neutral"] = 1.0

        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]

        summary = self._generate_psychological_summary(emotions, au_features, micro_expressions)

        composite_emotions = [k for k, v in emotions.items() if v > 0.3 and k not in [
            "happy", "sadness", "anger", "fear", "surprise", "disgust"
        ]]

        return EmotionResult(
            emotion_vector=emotions,
            dominant_emotion=dominant,
            confidence=round(confidence, 3),
            composite_emotions=composite_emotions,
            psychological_summary=summary
        )

    def _generate_psychological_summary(self, emotions, au_features, micro_expressions):
        summary_parts = []

        dominant = max(emotions, key=emotions.get)
        conf = emotions[dominant]
        if conf > 0.5:
            summary_parts.append(f"主导情绪为{dominant}（置信度{conf:.2f}）")

        composite = [k for k in ["forced_smile", "contempt", "cognitive_load", "moral_disgust"] if emotions.get(k, 0) > 0.3]
        if composite:
            summary_parts.append("检测到复合情绪：" + "、".join(composite))

        if micro_expressions and any(micro_expressions.data.values()):
            summary_parts.append("发现微表情活动")

        tension = getattr(au_features, 'psychological_signals', {}).get('tension_score', 0)
        if tension > 0.6:
            summary_parts.append("处于高紧张状态")
        elif tension > 0.3:
            summary_parts.append("有中等紧张表现")

        if not summary_parts:
            return "当前情绪状态平稳"
        return "；".join(summary_parts) + "。"
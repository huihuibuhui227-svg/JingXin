# tests/test_gesture_dead_code_removed.py
"""M3.0 Task 9:手势那 4 个死代码 extractor 与 `GesturePipeline` 必须**真删干净**。

**为什么它们该死**:设计文档 `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md`
对 `hand_feature_extractor.py` / `arm_feature_extractor.py` / `shoulder_feature_extractor.py` /
`upper_body_feature_extractor.py` 的处置是「删代码 —— 不在 CSV 生产路径上(死代码),
与生效 analyzer 取值不同」。CSV 生产路径走的是 `gesture_analysis/core/analysis/*_analyzer.py`
(见 `experiments/extract_features.py` 与 `gesture_analysis/api/app.py`);
`gesture_analysis/pipeline/gesture_pipeline.py` 则没有任何 `.py` 生产代码 import 它、
也没有任何测试碰它。本文件守的是「删干净了,而且包还能 import」。

**红法(两种,做本任务时逐条实测过)**:

① **只删文件、不清导入**:删掉那 4 个 `*_feature_extractor.py` 而不改
   `gesture_analysis/core/feature_extraction/__init__.py` 里那 4 行
   `from .<name>_feature_extractor import <Name>FeatureExtractor`
   ⟹ `import gesture_analysis` 直接 `ModuleNotFoundError`。
   根因是 import 链:`gesture_analysis/__init__.py` → `gesture_analysis/core/__init__.py`
   → `gesture_analysis/core/feature_extraction/__init__.py` → 被删的模块。
   所以本文件的 `test_gesture_package_imports_without_the_dead_extractors` 那一刻**直接炸**。

② **悬空名字**:文件删了、上面那条 import 也删了,但 `gesture_analysis/__init__.py`
   的 `__all__` 里那 4 个名字还挂着 —— 那个文件**从来没有 import 过它们**
   (`__all__` 与 `import` 是分开写的),所以它们是**悬空名字**:
   `from gesture_analysis import *` 本来就会 `AttributeError`。
   此时 ①的 import 不会炸,但第 3 条断言(`gone not in gesture_analysis.__all__`)红。

**★ 2026-09-26 M3.0 整支复核(Important 5)把「悬空名字」从登记升级成断言**:那时
`gesture_analysis.__all__` 里还有 **17 个没有对应 import 的名字**,`from gesture_analysis import *`
**实测直接炸**。其中:

- **5 个是真类**(`HandFeatures` / `ShoulderFeatures` / `ArmFeatures` / `UpperBodyFeatures` /
  `EmotionResult`)—— 由 `gesture_analysis/models/__init__.py:__all__` 导出(单一真源),
  缺的只是本文件那行 import ⟹ **补 import 收进来**(处置是"收",不是"删");
- **12 个全仓不存在**(`HandEmotionAnalyzer` / `GestureEmotionPipeline` …)⟹ **删名字**。

原文只在 docstring 里写「本文件不替它们背书」—— **那句话没有任何约束力**,拦不住第 18 个
悬空名字被加进去。现在由下面两条断言守着(其中一条**不依赖名单**:它问的是"`__all__` 里
每个名字是不是真的在模块上")。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 被删的 4 个类:必须既不在导出表里,也不在 `gesture_analysis.__all__` 里。
DEAD_EXTRACTORS = (
    "HandFeatureExtractor",
    "ArmFeatureExtractor",
    "ShoulderFeatureExtractor",
    "UpperBodyFeatureExtractor",
)

# 被删的 5 个模块文件(相对仓库根)。
DEAD_MODULES = (
    "gesture_analysis/core/feature_extraction/hand_feature_extractor.py",
    "gesture_analysis/core/feature_extraction/arm_feature_extractor.py",
    "gesture_analysis/core/feature_extraction/shoulder_feature_extractor.py",
    "gesture_analysis/core/feature_extraction/upper_body_feature_extractor.py",
    "gesture_analysis/pipeline/gesture_pipeline.py",
)


def test_dead_code_files_are_really_gone():
    """红法:把 `DEAD_MODULES` 里任何一个文件(或它的内容)还原回工作树。

    这条是 Task 9「断链核查」的机器版:文件在 ⟹ 说明删除没做全,
    或者被谁改回来了。
    """
    survivors = [rel for rel in DEAD_MODULES if (ROOT / rel).exists()]
    assert survivors == [], f"这些死代码文件还在工作树里:{survivors}"


def test_gesture_package_imports_without_the_dead_extractors():
    """红法①:删文件但不清 `__init__.py` 的 import ⟹ 本函数在 import 那一刻就炸。
    红法②:import 清了但 `gesture_analysis.__all__` 里的悬空名字没清 ⟹ 第 3 条断言红。
    红法③(2026-09-26 Task 10 补):`__all__` 里**只有名字、没有对应的 import** ⟹
      同上一条,`from ... import *` 会 `AttributeError`。
    """
    import gesture_analysis
    import gesture_analysis.core as core
    import gesture_analysis.core.feature_extraction as fe

    for gone in DEAD_EXTRACTORS:
        assert not hasattr(fe, gone), (
            f"{gone} 还在 gesture_analysis.core.feature_extraction 的导出表里 —— "
            f"它的模块文件已经删了,这个名字现在只能来自 feature_extraction/__init__.py 的残留 import"
        )
        assert not hasattr(core, gone), (
            f"{gone} 还在 gesture_analysis.core 的导出表里 —— "
            f"core/__init__.py 的 `from .feature_extraction import (...)` 没清干净"
        )
        assert gone not in getattr(fe, "__all__", ()), (
            f"{gone} 还挂在 gesture_analysis.core.feature_extraction.__all__ 上 —— "
            f"`__all__` 与 import 是分开写的,把名字放回去而**不**恢复 import 时,"
            f"`hasattr` 是 False(所以上面两条查不出来),而 "
            f"`from gesture_analysis.core.feature_extraction import *` 会 AttributeError"
        )
        assert gone not in getattr(core, "__all__", ()), (
            f"{gone} 还挂在 gesture_analysis.core.__all__ 上 —— "
            f"同上:`core/__init__.py` 里 `__all__` 与 import 是分开写的,"
            f"只把名字放回 `__all__` 不会让 `hasattr` 为真,但 `from gesture_analysis.core import *` 会炸"
        )
        assert gone not in gesture_analysis.__all__, (
            f"{gone} 还挂在 gesture_analysis.__all__ 上 —— "
            f"而那个文件根本没有 import 过它,这是个悬空名字(会把 `from gesture_analysis import *` 弄炸)"
        )


def test_gesture_pipeline_is_gone():
    """红法:还原 `gesture_analysis/pipeline/gesture_pipeline.py`
    并还原 `gesture_analysis/pipeline/__init__.py` 里那行
    `from .gesture_pipeline import GesturePipeline`(以及 `__all__` 里的名字)。
    """
    import gesture_analysis.pipeline as pipeline

    assert not hasattr(pipeline, "GesturePipeline"), (
        "GesturePipeline 还在 gesture_analysis.pipeline 的导出表里 —— "
        "它的模块文件已经删了,这个名字现在只能来自 pipeline/__init__.py 的残留 import"
    )
    assert "GesturePipeline" not in pipeline.__all__, (
        "GesturePipeline 还挂在 gesture_analysis.pipeline.__all__ 上(悬空名字)"
    )


# ── 悬空名字(2026-09-26 M3.0 整支复核 Important 5)────────────────────────
# 12 个**全仓不存在**的名字:既没有任何定义,也没有对应 import ⟹ 已从 `__all__` 删掉。
DANGLING_NAMES_REMOVED = (
    "HandEmotionAnalyzer", "ShoulderEmotionAnalyzer", "ArmEmotionAnalyzer",
    "UpperBodyEmotionAnalyzer", "EmotionFusionAnalyzer", "GestureFeatures",
    "HandEmotionResult", "ShoulderEmotionResult", "ArmEmotionResult",
    "UpperBodyEmotionResult", "GestureEmotionResult", "GestureEmotionPipeline",
)

# 5 个**真类**:由 `gesture_analysis/models/__init__.py:__all__` 导出 ⟹ 补 import 收进来后
# 必须仍然留在 `__all__` 里(它们曾经也是悬空名字 —— 缺的是 import,不是名字)。
REAL_MODELS_EXPORTED = (
    "HandFeatures", "ShoulderFeatures", "ArmFeatures", "UpperBodyFeatures", "EmotionResult",
)


def test_gesture_all_has_no_dangling_names():
    """★ 红法(两条,都是生产改动):
      · 往 `gesture_analysis/__init__.py` 的 `__all__` 里写回 `"HandEmotionAnalyzer"`
        (或其馀 11 个全仓不存在的名字)⟹ 第 1 条断言红;
      · 往 `__all__` 里加**任何一个没有对应 import 的名字**(例如 `"Foo"`)⟹ 第 2 条断言红。
        ★ **这一条才是拦住"第 18 个悬空名字"的那道闸 —— 它不依赖上面那张名单。**

    为什么必须问 `hasattr` 而不是只查名单:`__all__` 与 `import` 是**分开写的两处**,
    名字挂在 `__all__` 里而 import 没了(或从来没有)时 `hasattr` 是 False,
    只有 `from ... import *` 会炸 —— 所以判据要直接问"这个名字真的在模块上吗"。
    """
    import gesture_analysis as pkg

    for gone in DANGLING_NAMES_REMOVED:
        assert gone not in pkg.__all__, (
            f"{gone} 又回到了 gesture_analysis.__all__ 上 —— 它**全仓没有定义**"
            f"(没有 import、也没有任何类/函数叫这个名字),`from gesture_analysis import *` 会 AttributeError")
    dangling = sorted(n for n in pkg.__all__ if not hasattr(pkg, n))
    assert dangling == [], (
        f"gesture_analysis.__all__ 里有**悬空名字**(挂了名字却没有对应对象):{dangling} —— "
        f"`__all__` 与 import 是分开写的两处,加名字时忘了加 import 就是这个形态;"
        f"要么补 import,要么把名字删掉")
    for real in REAL_MODELS_EXPORTED:
        assert real in pkg.__all__ and hasattr(pkg, real), (
            f"{real} 应当留在 gesture_analysis.__all__ 里(它是 `gesture_analysis.models` 导出的真类,"
            f"而且顶层再导出一次是这次修复的一部分)")


def test_star_import_from_gesture_analysis_works():
    """★ 红法:往 `__all__` 里加一个没有 import 的名字 ⟹ 下面那句 `exec` 当场 `AttributeError`。

    这是**端到端**的那一条:它不解析 `__all__`,而是真的执行 `from gesture_analysis import *` ——
    2026-09-26 整支复核实测:改前这一句直接炸(17 个悬空名字),改后可用。
    """
    ns = {}
    exec("from gesture_analysis import *", ns)          # noqa: S102 —— 要的就是执行这一句
    for n in ("HandFeatures", "ShoulderFeatures", "ArmFeatures", "UpperBodyFeatures",
              "EmotionResult", "HandAnalyzer", "ShoulderAnalyzer", "ArmAnalyzer",
              "UpperBodyAnalyzer", "EmotionInferencer"):
        assert n in ns, f"`from gesture_analysis import *` 没带出 {n}"

## Task 1: 测试脚手架

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: 无
- Produces: 可用的 pytest 环境;`tests/` 下所有测试可 `from report_frontend.X import Y`

- [ ] **Step 1: 建分支**

```bash
cd /home/huihuibuhui/jingxin
git checkout -b fix/report-layer-evidence-gate
```

- [ ] **Step 2: 写 conftest.py(把项目根加入 sys.path)**

```python
# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: 写冒烟测试**

```python
# tests/test_smoke.py
def test_package_imports():
    """report_frontend 包可导入(会连带导入 visualizer → 需要 plotly)。"""
    import report_frontend

    assert report_frontend.ResearchCapabilityMapper is not None
    assert report_frontend.ReportGenerator is not None
```

- [ ] **Step 4: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_smoke.py -v`
Expected: PASS。若报 `ModuleNotFoundError: plotly`,说明用错了解释器。

- [ ] **Step 5: 提交**

```bash
git add tests/conftest.py tests/test_smoke.py
git commit -m "test: 为报告层引入 pytest 脚手架"
```

---


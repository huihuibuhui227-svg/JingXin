# tests/test_smoke.py
def test_package_imports():
    """report_frontend 包可导入(会连带导入 visualizer → 需要 plotly)。"""
    import report_frontend

    assert report_frontend.ResearchCapabilityMapper is not None
    assert report_frontend.ReportGenerator is not None

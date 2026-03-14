# main/api/app.py
"""
集成 API 入口（预留）
当前可为空，或简单返回健康检查
"""
from fastapi import FastAPI

app = FastAPI(title="JingXin Integrated API")

@app.get("/health")
def health():
    return {"status": "ok", "message": "JingXin main module ready"}
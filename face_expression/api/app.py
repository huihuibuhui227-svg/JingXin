from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os

# 使用标准导入（假设项目已正确安装或在 PYTHONPATH 中）
try:
    from face_expression import StaticFaceAnalyzer
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已正确安装 face_expression 模块")
    raise

app = FastAPI(
    title="Face Expression Analysis API",
    description="面部表情分析API，支持图片上传分析"
)

# 初始化分析器（延迟初始化，只有在首次请求时才加载 MediaPipe）
analyzer = None

def get_analyzer():
    """获取分析器实例（延迟初始化）"""
    global analyzer
    if analyzer is None:
        try:
            analyzer = StaticFaceAnalyzer()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"分析器初始化失败: {str(e)}")
    return analyzer

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Face Expression Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - 上传图片进行AU分析",
            "/health": "GET - 健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    上传图片进行AU分析

    参数:
        file: 上传的图片文件

    返回:
        JSON格式的分析结果，包含AU特征和情绪
    """
    # 检查文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传的文件必须是图片格式")

    try:
        # 创建临时文件保存上传的图片
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # 获取分析器并分析图片
        analyzer = get_analyzer()
        features = analyzer.analyze_image(temp_path)

        # 删除临时文件
        os.unlink(temp_path)

        if features is None:
            raise HTTPException(status_code=400, detail="无法分析图片，请确保图片包含清晰的正脸")

        # 返回分析结果
        return JSONResponse(content={
            "status": "success",
            "result": features
        })

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
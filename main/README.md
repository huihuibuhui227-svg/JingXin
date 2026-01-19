# JingXin 集成模块

## 模块简介

JingXin集成模块是整个项目的核心协调器，它将面部表情分析、语音交互和手势分析三个子模块整合在一起，提供统一的接口和协调机制，实现多模态数据的融合分析。

## 模块结构

```
jingxin/
├── __init__.py              # 模块初始化文件
├── README.md               # 模块说明文档
├── integrator.py           # 集成器
├── api/                   # API模块
│   ├── __init__.py
│   └── app.py            # FastAPI应用
└── examples/              # 示例脚本
    ├── __init__.py
    └── run_integrated_interview.py  # 集成面试示例
```

## 功能特性

### 1. 统一接口
- 提供统一的API访问三个子模块
- 简化多模块协同调用
- 统一的数据格式和错误处理

### 2. 面试流程管理
- 管理完整的面试会话
- 协调问题提出和回答收集
- 记录面试过程中的多模态数据

### 3. 多模态数据融合
- 同时分析面部表情、语音和手势
- 综合评估候选人的情绪状态
- 生成全面的评估报告

### 4. Web API
- 提供RESTful API接口
- 支持实时分析
- 便于集成到其他系统

## 使用方法

### 1. 基本使用

```python
from main import JingXinIntegrator

# 初始化集成器
integrator = JingXinIntegrator()

# 开始面试会话
integrator.start_interview_session()

# 提出问题
question = integrator.ask_question()

# 获取回答
answer = integrator.get_answer()

# 分析视频帧
results = integrator.analyze_frame(frame)

# 获取综合评估
evaluation = integrator.get_comprehensive_evaluation()

# 结束面试
evaluation = integrator.end_interview_session()
```

### 2. 运行示例脚本

```bash
# 运行集成面试
python main/examples/run_integrated_interview.py
```

### 3. 使用API

```bash
# 启动API服务
python main/api/app.py
```

API端点：
- `GET /` - API信息
- `GET /health` - 健康检查
- `POST /interview/start` - 开始面试
- `GET /interview/question` - 获取下一个问题
- `POST /interview/answer` - 提交回答
- `GET /interview/evaluation` - 获取综合评估
- `POST /analyze/frame` - 分析视频帧
- `POST /tts` - 文本转语音

## API接口说明

### 面试接口

#### 开始面试
```http
POST /interview/start
```

#### 获取问题
```http
GET /interview/question
```

#### 提交回答
```http
POST /interview/answer
Content-Type: application/json

{
  "answer": "回答内容"
}
```

#### 获取评估
```http
GET /interview/evaluation
```

### 分析接口

#### 分析视频帧
```http
POST /analyze/frame
Content-Type: application/json

{
  "image": "base64编码的图片"
}
```

返回示例：
```json
{
  "status": "success",
  "results": {
    "timestamp": 1234567890.123,
    "face": {
      "emotion": "happy",
      "confidence": 0.95
    },
    "gesture": {
      "emotion_state": "放松",
      "overall_score": 85.5,
      "feedback": "你看起来很放松，状态很棒！"
    }
  }
}
```

### 语音接口

#### 文本转语音
```http
POST /tts
Content-Type: application/json

{
  "text": "要朗读的文本"
}
```

## 依赖要求

集成模块依赖三个子模块，确保已安装所有子模块的依赖。

## 注意事项

1. 首次使用时，集成器会自动初始化三个子模块
2. 如果某个子模块初始化失败，集成器会继续运行，但相关功能将不可用
3. 建议在使用前检查各模块的健康状态
4. 视频帧分析需要传入BGR格式的OpenCV图片

## 许可证

本模块是JingXin项目的一部分，遵循项目的许可证协议。

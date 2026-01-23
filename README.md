# JingXin 多模态面试评估系统

## 项目简介

JingXin是一个基于多模态AI技术的智能面试评估系统，通过同时分析候选人的面部表情、手势姿态和语音内容，为面试官提供全面、客观的评估报告。系统采用模块化设计，包含面部微表情分析、手势与肢体语言分析、语音交互分析三大核心模块，以及一个集成模块用于协调各模块工作。

## 项目特色

- **多模态融合分析**：同时分析面部表情、手势姿态和语音内容，提供全方位评估
- **实时检测与反馈**：支持实时视频流分析，即时提供情绪状态和反馈建议
- **模块化架构**：各功能模块独立开发、测试和部署，便于维护和扩展
- **RESTful API**：提供标准API接口，便于集成到现有系统
- **可视化展示**：直观展示分析结果，支持中文显示

## 项目结构

```
jingxin/
├── data/                           # 数据目录
│   ├── input/                      # 输入数据
│   └── output/                     # 输出数据
├── face_expression/                # 面部微表情分析模块
│   ├── analyzers/                  # 分析器
│   │   ├── emotion_engine.py       # 情绪识别引擎
│   │   ├── face_au_analyzer.py     # 实时视频分析器
│   │   ├── feature_extractor.py    # 特征提取器
│   │   ├── image_analyzer.py       # 静态图片分析器
│   │   ├── landmarks.py            # 面部关键点索引配置
│   │   ├── micro_expression.py     # 微表情分析
│   │   └── tension_engine.py       # 紧张度分析引擎
│   ├── api/                        # API接口
│   │   └── app.py                  # FastAPI应用
│   ├── examples/                   # 示例脚本
│   │   ├── run_image_analyzer.py   # 图片分析示例
│   │   └── run_video_analyzer.py   # 视频分析示例
│   ├── inference/                  # 推理模块
│   │   └── emotion_infer.py        # 情绪推断引擎
│   ├── utils/                      # 工具模块
│   │   ├── logger.py               # 日志工具
│   │   └── visualize.py            # 可视化工具
│   ├── config.py                   # 配置文件
│   ├── MODULE_DEPENDENCIES.md      # 模块依赖文档
│   └── README.md                   # 模块说明文档
├── gesture_analysis/               # 手势与肢体语言分析模块
│   ├── analyzers/                  # 分析器
│   │   ├── hand_analyzer.py        # 手部分析器
│   │   └── shoulder_analyzer.py    # 肩部分析器
│   ├── api/                        # API接口
│   │   └── app.py                  # FastAPI应用
│   ├── examples/                   # 示例脚本
│   │   ├── run_realtime_analyzer.py # 实时分析示例
│   │   └── visualize_logs.py       # 日志可视化
│   ├── inference/                  # 推理模块
│   │   └── emotion_inferencer.py   # 情绪推断器
│   ├── utils/                      # 工具模块
│   │   ├── logger.py               # 日志工具
│   │   ├── visualization.py        # 可视化工具
│   │   └── visualize.py            # 可视化工具
│   ├── config.py                   # 配置文件
│   └── README.md                   # 模块说明文档
├── voice_interaction/              # 语音交互分析模块
│   ├── analyzers/                  # 分析器
│   │   ├── prosody_analyzer.py     # 韵律分析器
│   │   ├── speech_recognizer.py    # 语音识别器
│   │   └── tts_engine.py           # 文本转语音引擎
│   ├── assessment/                 # 评估模块
│   │   ├── interview_assessment.py # 面试评估
│   │   └── research_assessment.py  # 科研评估
│   ├── api/                        # API接口
│   │   └── app.py                  # FastAPI应用
│   ├── examples/                   # 示例脚本
│   │   ├── run_interview.py        # 面试示例
│   │   ├── run_research_assessment.py # 科研评估示例
│   │   └── visualize_log.py        # 日志可视化
│   ├── utils/                      # 工具模块
│   │   ├── logger.py               # 日志工具
│   │   └── visualize.py            # 可视化工具
│   ├── config.py                   # 配置文件
│   └── __init__.py                 # 模块初始化文件
├── main/                           # 集成模块
│   ├── api/                        # API接口
│   │   └── app.py                  # FastAPI应用
│   ├── examples/                   # 示例脚本
│   │   └── run_integrated_interview.py # 集成面试示例
│   ├── integrator.py               # 集成器
│   └── README.md                   # 模块说明文档
└── requirements.txt                # 项目依赖
```

## 核心功能

### 1. 面部微表情分析模块

基于MediaPipe的面部动作单元(AU)分析和情绪识别功能，支持实时视频流和静态图片两种分析模式。

**主要功能：**
- 实时检测面部关键点
- 计算眨眼频率
- 提取多个AU特征值
- 实时情绪识别
- 疲劳检测
- 专注度评估

**支持识别的情绪状态：**
- 😊 愉悦
- 🧠 专注
- ❓ 困惑
- 😮 惊讶
- 🤢 厌恶
- 😢 悲伤
- 💬 说话
- 😴 疲劳（仅视频模式）
- 😐 中性

### 2. 手势与肢体语言分析模块

提供实时手势检测、肩部动作分析和情绪评估功能，用于评估用户的抗压能力和情绪状态。

**主要功能：**
- 手部分析：
  - 实时检测手部关键点
  - 计算手指抖动幅度
  - 检测握拳状态
  - 计算手指张开度
  - 评估抗压能力评分

- 肩部分析：
  - 实时检测肩部位置
  - 自动校准自然肩位
  - 计算肩部抖动幅度
  - 检测耸肩动作
  - 评估紧张度评分

- 情绪评估：
  - 综合手部和肩部特征
  - 推断情绪状态（非常放松、放松、中性、轻微紧张、紧张、高度焦虑）
  - 提供实时反馈和建议

### 3. 语音交互分析模块

提供语音识别、文本转语音和面试评估功能，用于分析候选人的语音内容和表达方式。

**主要功能：**
- 语音识别：
  - 实时语音转文字
  - 支持中文识别
  - 自动检测语音停顿

- 文本转语音：
  - 将问题文本转换为语音
  - 支持语速和音量调节

- 面试评估：
  - 预设面试问题库
  - 分析回答内容
  - 评估核心胜任力与品质
  - 分析语音表达表现

### 4. 集成模块

统一调用面部表情、手势姿态、语音内容三个子模块，实现多模态数据的融合分析。

**主要功能：**
- 统一接口：提供统一的API访问三个子模块
- 面试流程管理：管理完整的面试会话，协调问题提出和回答收集
- 多模态数据融合：同时分析面部表情、语音和手势，综合评估候选人的情绪状态
- Web API：提供RESTful API接口，支持实时分析，便于集成到其他系统

## 快速开始

### 环境要求

- Python 3.11
- Windows/Linux/macOS
- 摄像头设备
- 麦克风设备

### 安装步骤

1. 克隆项目仓库：
```bash
git clone <repository-url>
cd jingxin
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载Vosk语音识别模型：
```bash
# 下载中文小模型（约40MB）
# 请从 https://alphacephei.com/vosk/models 下载
# 将模型解压到项目根目录，命名为 vosk-model-small-cn-0.22
```

5. 配置API密钥（可选）：
```bash
# 创建 .env 文件
echo "DASHSCOPE_API_KEY=your_api_key" > .env
```

### 运行示例

#### 运行面部表情分析
```bash
python face_expression/examples/run_video_analyzer.py
```

#### 运行手势分析
```bash
python gesture_analysis/examples/run_realtime_analyzer.py
```

#### 运行语音面试
```bash
python voice_interaction/examples/run_interview.py
```

#### 运行集成面试
```bash
python main/examples/run_integrated_interview.py
```

### 启动API服务

#### 启动面部表情分析API
```bash
python face_expression/api/app.py
```

#### 启动手势分析API
```bash
python gesture_analysis/api/app.py
```

#### 启动语音交互API
```bash
python voice_interaction/api/app.py
```

#### 启动集成API
```bash
python main/api/app.py
```

## API文档

### 集成模块API端点

- `GET /` - API信息
- `GET /health` - 健康检查
- `POST /interview/start` - 开始面试
- `GET /interview/question` - 获取下一个问题
- `POST /interview/answer` - 提交回答
- `GET /interview/evaluation` - 获取综合评估
- `POST /analyze/frame` - 分析视频帧
- `POST /tts` - 文本转语音

### 面部表情分析API端点

- `GET /` - API信息
- `GET /health` - 健康检查
- `POST /analyze/image` - 分析静态图片
- `POST /analyze/video` - 分析视频流

### 手势分析API端点

- `GET /` - API信息
- `GET /health` - 健康检查
- `POST /analyze` - 分析图片中的手势和肩部
- `POST /reset` - 重置分析器状态

### 语音交互API端点

- `GET /` - API信息
- `GET /health` - 健康检查
- `POST /interview/start` - 开始面试
- `GET /interview/question` - 获取下一个问题
- `POST /interview/answer` - 提交回答
- `GET /interview/evaluation` - 获取评估报告
- `POST /tts` - 文本转语音

## 技术栈

- **计算机视觉**：MediaPipe, OpenCV
- **深度学习**：TensorFlow, PyTorch
- **语音处理**：Vosk, librosa, sounddevice, pyttsx3
- **Web框架**：FastAPI, Uvicorn
- **数据处理**：NumPy, Pandas, SciPy
- **可视化**：Matplotlib
- **AI服务**：DashScope (通义千问)

## 开发指南

### 模块开发规范

1. 每个模块应包含以下目录结构：
   - `analyzers/` - 核心分析逻辑
   - `api/` - API接口
   - `examples/` - 使用示例
   - `utils/` - 工具函数
   - `config.py` - 配置文件
   - `README.md` - 模块文档

2. 每个模块应提供独立的API接口，便于单独测试和部署

3. 模块间通信应通过集成模块进行，避免直接依赖

### 代码风格

- 遵循PEP 8代码风格
- 使用类型注解
- 编写清晰的文档字符串
- 添加必要的注释

### 测试

- 每个模块应包含单元测试
- 集成测试应覆盖主要功能流程
- 使用pytest进行测试

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至：[your-email@example.com]

## 致谢

感谢以下开源项目：
- MediaPipe
- OpenCV
- Vosk
- FastAPI
- 通义千问 (DashScope)

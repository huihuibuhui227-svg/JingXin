# JingXin 多模态面试评估系统

## 概述

JingXin是一个智能多模态面试评估系统，通过同时分析面试者的面部表情、手势姿态和语音内容，提供全面、客观的面试评估报告。系统采用先进的计算机视觉和语音处理技术，能够实时监测和分析面试者的情绪状态、紧张程度和表达能力。

## 核心特性

### 多模态分析
- **面部表情分析**: 检测面部动作单元，识别情绪状态和专注度
- **手势姿态分析**: 评估手部动作、肩部稳定性、手臂角度
- **语音交互分析**: 分析音调、能量、流畅度等语音特征

### 智能评估
- **面试评估**: 针对面试场景的综合能力评估
- **科研评估**: 针对科研场景的专业能力评估
- **情绪融合**: 多模态情绪状态综合评估
- **实时反馈**: 提供即时的分析结果和建议

### 数据管理
- **结构化日志**: CSV和JSON格式的详细日志记录
- **可视化报告**: 热力图、统计图表等可视化输出
- **历史追踪**: 支持历史数据对比和趋势分析

## 系统架构

```
jingxin/
├── face_expression/        # 面部表情分析模块
│   ├── core/             # 核心分析引擎
│   ├── models/           # 数据模型
│   ├── pipeline/         # 处理流程
│   ├── utils/           # 工具函数
│   └── examples/        # 示例代码
├── gesture_analysis/      # 手势姿态分析模块
│   ├── core/             # 核心分析引擎
│   ├── models/           # 数据模型
│   ├── pipeline/         # 处理流程
│   ├── utils/           # 工具函数
│   └── examples/        # 示例代码
├── voice_interaction/     # 语音交互分析模块
│   ├── core/             # 核心分析引擎
│   ├── models/           # 数据模型
│   ├── pipeline/         # 处理流程
│   ├── utils/           # 工具函数
│   └── examples/        # 示例代码
├── main/                # 多模态集成模块
│   ├── api/             # 集成API
│   ├── examples/        # 集成示例
│   └── integrator.py    # 集成器
├── data/                # 数据目录
│   ├── input/           # 输入数据
│   ├── output/          # 输出结果
│   │   ├── face_expression/
│   │   ├── gesture_analysis/
│   │   └── voice_interaction/
│   └── logs/            # 日志文件
└── requirements.txt      # 依赖列表
```

## 安装指南

### 环境要求
- Python 3.8+
- Windows/Linux/macOS
- 摄像头
- 麦克风

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/yourusername/jingxin.git
cd jingxin
```

2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scriptsctivate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 下载语音模型
```bash
# 下载Vosk中文模型
cd vosk-model-cn-0.22
# 模型文件已包含在项目中
```

## 快速开始

### 1. 面部表情分析

```python
from face_expression import VideoPipeline
import cv2

pipeline = VideoPipeline()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = pipeline.process_frame(frame)
    print(f"情绪: {result.emotion}, 专注度: {result.features.focus_score}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 2. 手势姿态分析

```python
from gesture_analysis import GestureEmotionPipeline
import cv2

pipeline = GestureEmotionPipeline()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pipeline.process_frame(frame)
    emotion_result = pipeline.get_emotion_result()
    print(f"情绪: {emotion_result['emotion_state']}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 3. 语音交互分析

```python
from voice_interaction import InterviewAssessmentPipeline

pipeline = InterviewAssessmentPipeline()

# 添加回答
pipeline.add_answer("我是计算机科学专业的学生...")

# 获取评估结果
evaluation = pipeline.get_comprehensive_evaluation()
print(evaluation)

# 保存日志
log_path = pipeline.save_log()
```

### 4. 多模态集成分析

```python
from main import JingXinIntegrator

integrator = JingXinIntegrator()
integrator.start_interview_session()
```

## API服务

### 启动API服务

```bash
# 面部表情API
python -m face_expression.api.app

# 手势分析API
python -m gesture_analysis.api.app

# 语音交互API
python -m voice_interaction.api.app

# 集成API
python -m main.api.app
```

### API端点

#### 面部表情API
- `POST /analyze/image`: 分析单张图片
- `POST /analyze/video`: 分析视频文件
- `GET /health`: 健康检查

#### 手势分析API
- `POST /analyze/frame`: 分析单帧图像
- `POST /analyze/video`: 分析视频流
- `GET /health`: 健康检查

#### 语音交互API
- `POST /recognize`: 语音识别
- `POST /synthesize`: 语音合成
- `POST /assess/interview`: 面试评估
- `POST /assess/research`: 科研评估
- `POST /visualize`: 数据可视化
- `GET /health`: 健康检查

## 数据目录结构

```
data/
├── input/              # 输入数据
│   ├── images/        # 图片文件
│   └── videos/        # 视频文件
├── output/            # 输出结果
│   ├── face_expression/     # 面部表情输出
│   ├── gesture_analysis/    # 手势分析输出
│   └── voice_interaction/  # 语音交互输出
└── logs/              # 日志文件
    ├── face_au_log.csv
    ├── static_face_log.csv
    ├── gesture_emotion_log_*.csv
    ├── interview_emotion_log_*.csv
    └── research_emotion_log_*.csv
```

## 配置说明

### 环境变量

```bash
# 自定义数据目录
export JINGXIN_DATA_DIR=/custom/path/to/data
```

### 路径配置

所有模块统一使用以下路径结构：
- `DATA_DIR`: 数据根目录
- `INPUT_DIR`: 输入文件目录
- `OUTPUT_DIR`: 输出结果目录
- `LOGS_DIR`: 日志文件目录

### 各模块输出目录

- `face_expression`: `data/output/face_expression`
- `gesture_analysis`: `data/output/gesture_analysis`
- `voice_interaction`: `data/output/voice_interaction`

## 使用场景

### 1. 面试评估
适用于企业招聘、学术面试等场景，评估面试者的：
- 核心胜任力
- 问题解决能力
- 团队合作意识
- 表达流畅度
- 情绪稳定性

### 2. 科研评估
适用于研究生入学、科研项目评估等场景，评估：
- 方法论能力
- 批判性思维
- 创新能力
- 可行性评估
- 坚持与韧性

### 3. 情绪监测
适用于心理辅导、情绪管理培训等场景，监测：
- 情绪状态变化
- 紧张程度
- 专注度水平
- 疲劳状态

## 性能优化

### 硬件加速
- 使用GPU加速MediaPipe推理
- 使用GPU加速librosa特征提取

### 参数调整
- 降低视频分辨率以提升处理速度
- 调整检测置信度阈值平衡准确率和速度
- 减少历史数据长度以降低内存使用

### 系统优化
- 使用多线程处理多模态数据
- 优化数据传输和存储
- 实现结果缓存机制

## 故障排除

### 摄像头问题
```bash
# 检查摄像头设备
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# 更换摄像头索引
cap = cv2.VideoCapture(1)  # 尝试索引1
```

### 麦克风问题
```bash
# 检查麦克风设备
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
```

### 模块导入问题
```bash
# 确保从项目根目录运行
cd /path/to/jingxin
python -m main.examples.run_integrated_interview
```

### 依赖问题
```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt

# 清理缓存
pip cache purge
```

## 项目贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 版本历史

### v1.0.0 (2024-01-01)
- 初始版本发布
- 实现基础多模态分析功能
- 支持面试和科研评估场景
- 提供API服务接口

## 许可证

Copyright © JingXin Team. All rights reserved.

## 联系方式

- 项目主页: https://github.com/yourusername/jingxin
- 问题反馈: https://github.com/yourusername/jingxin/issues
- 邮箱: contact@jingxin.team

## 致谢

感谢以下开源项目的支持：
- MediaPipe
- Vosk
- Librosa
- OpenCV
- FastAPI

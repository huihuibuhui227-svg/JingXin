# JingXin 多模态面试评估系统

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)

## 项目简介

JingXin是一个智能多模态面试评估系统，通过同时分析面试者的面部表情、手势姿态、语音内容和眼动轨迹，提供全面、客观的面试评估报告。系统采用先进的计算机视觉和语音处理技术，能够实时监测和分析面试者的情绪状态、紧张程度和表达能力，并生成专业的科研能力评估报告。

## 核心特性

### 多模态分析
- **面部表情分析**: 检测面部动作单元(AU)，识别情绪状态、微表情和紧张度
- **手势姿态分析**: 评估手部动作、肩部稳定性、手臂角度和肢体语言
- **语音交互分析**: 分析音调、能量、流畅度等语音特征
- **眼动追踪**: 分析注视轨迹、稳定性和眼动模式

### 智能评估
- **面试评估**: 针对面试场景的综合能力评估
- **科研评估**: 针对科研场景的专业能力评估，包含五个核心维度
  - 认知效率
  - 沟通流畅度
  - 自信水平
  - 逻辑思维
  - 压力韧性
- **情绪融合**: 多模态情绪状态综合评估
- **实时反馈**: 提供即时的分析结果和建议

### 数据管理
- **结构化日志**: CSV和JSON格式的详细日志记录
- **可视化报告**: 热力图、雷达图、轨迹图等可视化输出
- **HTML报告**: 自动生成交互式HTML评估报告
- **历史追踪**: 支持历史数据对比和趋势分析
- **数据库存储**: 支持SQL Server数据库存储

### Web界面
- **总控平台**: 统一的Web控制台，方便操作和查看结果
- **模块化运行**: 独立运行各分析模块
- **实时日志**: 在线查看运行日志和输出结果

## 系统架构

```
jingxin/
├── app.py                          # Flask Web应用主入口
├── templates/
│   └── dashboard.html              # Web总控台界面
├── face_expression/                # 面部表情分析模块
│   ├── core/
│   │   ├── analysis/              # 情绪、微表情、紧张度分析
│   │   └── feature_extraction/    # AU特征提取
│   ├── models/                   # 数据模型
│   ├── pipeline/                 # 处理流程
│   ├── utils/                   # 工具函数
│   └── examples/                # 示例代码
├── gesture_analysis/              # 手势姿态分析模块
│   ├── core/
│   │   ├── analysis/              # 手部、手臂、肩部、上身分析
│   │   └── feature_extraction/    # 特征提取
│   ├── models/                   # 数据模型
│   ├── pipeline/                 # 处理流程
│   ├── utils/                   # 工具函数
│   └── examples/                # 示例代码
├── voice_interaction/             # 语音交互分析模块
│   ├── core/
│   │   ├── analysis/              # 韵律分析
│   │   └── feature_extraction/    # 音频特征提取
│   ├── models/                   # 数据模型
│   ├── pipeline/                 # 处理流程
│   ├── utils/                   # 工具函数
│   └── examples/                # 示例代码
├── main/                        # 多模态集成模块
│   ├── api/                     # 集成API
│   ├── examples/                # 集成示例
│   ├── integrator.py            # 多模态集成器
│   └── storage.py              # 数据存储模块
├── report_frontend/              # 报告生成前端模块
│   ├── data_loader.py           # 数据加载器
│   ├── feature_engine.py        # 心理特征提取引擎
│   ├── research_mapper.py       # 科研能力映射器
│   ├── report_generator.py      # 报告生成器
│   └── visualizer.py          # 可视化工具
├── data/                        # 数据目录
│   ├── input/                  # 输入数据
│   ├── output/                 # 输出结果
│   └── logs/                  # 日志文件
├── vosk-model-cn-0.22/         # Vosk中文语音模型
└── requirements.txt              # 依赖列表
```

## 环境要求

- Python 3.8+
- Windows/Linux/macOS
- 摄像头
- 麦克风
- SQL Server (可选，用于数据库存储)

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/jingxin.git
cd jingxin

# 创建虚拟环境(推荐)
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动Web总控平台

```bash
# 运行主应用
python app.py
```

然后在浏览器中访问 http://127.0.0.1:5000

### 3. 使用各分析模块

通过Web界面或命令行运行各模块：

**面部表情分析**
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

**手势姿态分析**
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

**语音交互分析**
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

**多模态集成分析**
```python
from main import JingXinIntegrator

# 初始化集成器
integrator = JingXinIntegrator()

# 启动会话
integrator.start_session()

# 处理视频帧
frame_rgb = ...  # 从摄像头获取的RGB帧
integrator.process_video_frame(frame_rgb)

# 处理手势关键点
hand_lms, pose_lms = ...  # MediaPipe关键点
integrator.process_gesture_landmarks(hand_lms, pose_lms)

# 处理语音输入
audio_data = ...  # 音频数据或文本
integrator.process_voice_input(audio_data)

# 获取评估报告
report = integrator.get_assessment_report()
print(report)

# 结束会话
integrator.end_session()
```

### 4. 生成科研能力评估报告

```python
from report_frontend.report_generator import ReportGenerator

generator = ReportGenerator()
report_path = generator.generate_report()
```

## Web总控平台功能

### 主界面
- **面部与眼动分析**: 运行面部特征提取和眼动分析
- **肢体姿态分析**: 运行手势和姿态分析
- **语音交互评估**: 运行语音评估（Interview/Research场景）
- **生成综合判推报告**: 整合所有数据生成HTML评估报告

### API端点
- `GET /`: 访问总控台主页
- `GET /api/files/<folder_name>`: 获取输出目录下的文件列表
- `GET /output/<filename>`: 访问输出文件
- `POST /api/run/<module>`: 触发模块运行
  - `face`: 面部特征提取
  - `gesture`: 肢体分析
  - `voice`: 语音评估
  - `report`: 报告生成

## 数据目录结构

```
data/
├── input/                          # 输入数据
│   ├── images/                      # 图片文件
│   └── videos/                      # 视频文件
├── output/                         # 输出结果
│   ├── face_expression/              # 面部表情输出
│   │   ├── face_au_log_*.png       # AU分析图表
│   │   └── emotion_radar.png       # 情绪雷达图
│   ├── gesture_analysis/             # 手势分析输出
│   │   └── gesture_analysis_*.png  # 手势分析图表
│   ├── voice_interaction/           # 语音交互输出
│   │   ├── interview_analysis_*.png # 面试分析图表
│   │   └── heatmap.png            # 热力图
│   ├── Research_Assessment_Report_*.html  # 科研评估报告
│   ├── evidence_*.html             # 各维度证据报告
│   ├── radar_chart_*.html         # 雷达图
│   └── gaze_trajectory_*.html     # 眼动轨迹图
└── logs/                          # 日志文件
    ├── face_au_log.csv
    ├── static_face_log.csv
    ├── gesture_emotion_log_*.csv
    ├── interview_emotion_log_*.csv
    └── research_emotion_log_*.csv
```

## 科研能力评估维度

系统从多模态数据中提取特征，映射到以下五个核心维度：

### 1. 认知效率 (Cognitive Efficiency)
- 面部表情复杂度
- 眼动稳定性
- 微表情频率
- 注意力集中度

### 2. 沟通流畅度 (Communication Fluency)
- 语音流畅度
- 手势协调性
- 表达连贯性
- 停顿频率

### 3. 自信水平 (Confidence Level)
- 语调变化
- 音量稳定性
- 肢体开放度
- 眼神接触

### 4. 逻辑思维 (Logical Thinking)
- 论证结构
- 关键词密度
- 推理深度
- 批判性思维指标

### 5. 压力韧性 (Stress Resilience)
- 紧张度变化
- 恢复速度
- 情绪调节能力
- 抗压表现

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

### 数据库配置
修改`main/storage.py`中的数据库配置：
```python
config = {
    'host': 'localhost',
    'server': 'YOUR_SERVER_NAME',
    'port': '1433',
    'user': 'YOUR_USERNAME',
    'password': 'YOUR_PASSWORD',
    'database': 'jingxin',
    'charset': 'UTF-8'
}
```

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

## 技术栈

### Web框架
- Flask: Web应用框架
- Werkzeug: WSGI工具库

### 计算机视觉
- MediaPipe: 手势和姿态检测
- OpenCV: 图像处理和可视化

### 语音处理
- Vosk: 语音识别
- Librosa: 音频特征提取

### 数据处理
- NumPy: 数值计算
- Pandas: 数据处理

### 数据存储
- SQL Server: 结构化数据存储
- JSONL: 日志文件格式

### 可视化
- Matplotlib: 基础图表
- Seaborn: 统计图表
- Plotly: 交互式图表
- Pillow: 图像处理

### 并发处理
- Threading: 多线程处理
- Lock: 线程安全

## 项目贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 版本历史

### v2.0.0 (2024-03-14)
- 新增Web总控平台
- 新增科研能力评估报告生成
- 新增眼动追踪分析
- 新增五个核心评估维度
- 优化多模态数据融合
- 改进可视化报告

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
- Flask
- Plotly

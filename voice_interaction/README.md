# Voice Interaction Module

## 概述

语音交互模块，提供语音识别、语音合成和面试评估功能。该模块集成了语音特征提取、情绪分析和智能评估，能够实时分析面试者的语音特征，包括音调、能量、流畅度等，并生成综合评估报告。

## 功能特性

### 核心功能
- **语音识别**: 基于Vosk的实时语音识别
- **语音合成**: 文本转语音功能
- **韵律特征提取**: 提取音高、能量、语速、停顿等特征
- **语音分析**: 分析语调变化、音量、流畅度
- **面试评估**: 面试和科研场景的智能评估
- **数据可视化**: 生成热力图和统计图表

### 分析能力
- 音高特征分析（平均值、变化、趋势、方向）
- 能量特征分析（平均值、变化）
- 流畅度分析（语音比例、停顿频率）
- 停顿特征分析（平均停顿时间、最长停顿时间）
- 情绪状态识别
- 综合评分和反馈生成

## 模块结构

```
voice_interaction/
├── __init__.py                 # 模块初始化
├── config.py                   # 配置文件
├── api/
│   ├── __init__.py
│   └── app.py                # FastAPI接口
├── core/
│   ├── __init__.py
│   ├── feature_extraction/    # 特征提取
│   │   └── prosody_extractor.py
│   └── analysis/            # 分析器
│       └── prosody_analyzer.py
├── models/
│   ├── __init__.py
│   └── voice_models.py      # 数据模型
├── pipeline/
│   ├── __init__.py
│   ├── voice_pipeline.py      # 语音处理流程
│   ├── speech_recognition_pipeline.py  # 语音识别流程
│   ├── tts_pipeline.py       # 语音合成流程
│   └── assessment_pipeline.py # 评估流程
├── utils/
│   ├── __init__.py
│   ├── logger.py            # 日志工具
│   └── visualize.py         # 可视化工具
└── examples/
    ├── __init__.py
    ├── run_interview_assessment_voice.py
    ├── run_research_assessment_voice.py
    └── run_speech_recognition.py
```

## 安装依赖

```bash
pip install librosa numpy pandas matplotlib vosk pyaudio
```

## 配置说明

### 路径配置
- `DATA_DIR`: 数据根目录 (默认: `data`)
- `INPUT_DIR`: 输入文件目录 (默认: `data/input`)
- `OUTPUT_DIR`: 输出结果目录 (默认: `data/output`)
- `LOGS_DIR`: 日志文件目录 (默认: `data/logs`)
- `VOICE_OUTPUT_DIR`: 语音交互输出目录 (默认: `data/output/voice_interaction`)

### 语音特征提取配置
```python
ProsodyFeatureExtractor配置:
- sample_rate: 音频采样率 (默认: 16000)
- fmin: 最低音高频率 (默认: C2)
- fmax: 最高音高频率 (默认: C7)
```

### 语音分析配置
```python
ProsodyAnalyzer配置:
- 音高评估阈值:
  * 平缓: pitch_std < 20
  * 自然: 20 <= pitch_std <= 40
  * 丰富: pitch_std > 40

- 音量评估阈值:
  * 偏轻: energy_mean < 0.5
  * 适中: 0.5 <= energy_mean <= 0.8
  * 洪亮: energy_mean > 0.8

- 流畅度评估阈值:
  * 流畅: speech_ratio > 0.6
  * 较连贯: 0.3 < speech_ratio <= 0.6
  * 犹豫: speech_ratio <= 0.3

- 停顿频率评估阈值:
  * 较少: pause_frequency <= 5
  * 适中: 5 < pause_frequency <= 10
  * 频繁: pause_frequency > 10
```

### 评估配置
```python
InterviewAssessmentPipeline:
- 问题数量: 8个预设问题
- 评估维度:
  * 科研意识
  * 问题解决能力
  * 团队合作
  * 内在动机
  * 批判性思维

ResearchAssessmentPipeline:
- 问题数量: 8个预设问题
- 评估维度:
  * 方法论能力
  * 批判性思维
  * 创新能力
  * 可行性评估
  * 坚持与韧性
```

## 使用方法

### 语音特征提取

```python
from voice_interaction import ProsodyFeatureExtractor
import librosa

# 初始化特征提取器
extractor = ProsodyFeatureExtractor(sample_rate=16000)

# 加载音频
audio, sr = librosa.load('audio.wav', sr=16000)

# 提取特征
features = extractor.extract_all_features(audio)

print(f"音高平均值: {features['pitch_mean']}")
print(f"能量平均值: {features['energy_mean']}")
print(f"语音比例: {features['speech_ratio']}")
```

### 语音分析

```python
from voice_interaction import ProsodyAnalyzer

# 初始化分析器
analyzer = ProsodyAnalyzer()

# 分析特征
analysis = analyzer.analyze_all(features)

print(f"综合评分: {analysis['overall_score']}")
print(f"反馈: {analysis['feedback']}")
```

### 面试评估

```python
from voice_interaction import InterviewAssessmentPipeline

# 初始化评估管道
pipeline = InterviewAssessmentPipeline()

# 添加回答
pipeline.add_answer("我是计算机科学专业的学生...")

# 获取评估结果
evaluation = pipeline.get_comprehensive_evaluation()
print(evaluation)

# 保存日志
log_path = pipeline.save_log()
print(f"日志已保存至: {log_path}")
```

### 科研评估

```python
from voice_interaction import ResearchAssessmentPipeline

# 初始化评估管道
pipeline = ResearchAssessmentPipeline()

# 添加回答
pipeline.add_answer("我研究的是深度学习在图像识别中的应用...")

# 获取评估结果
evaluation = pipeline.get_comprehensive_evaluation()
print(evaluation)

# 保存日志
log_path = pipeline.save_log()
print(f"日志已保存至: {log_path}")
```

### 数据可视化

```python
from voice_interaction.utils.visualize import plot_heatmap

# 生成热力图
plot_heatmap(csv_path='data/logs/interview_emotion_log_20240101_120000.csv', log_type='interview')

# 生成面试评估图表
from voice_interaction.utils.visualize import plot_interview_logs
plot_interview_logs(csv_path='data/logs/interview_emotion_log_20240101_120000.csv')

# 生成科研评估图表
from voice_interaction.utils.visualize import plot_research_logs
plot_research_logs(csv_path='data/logs/research_emotion_log_20240101_120000.csv')
```

### 使用API服务

```bash
# 启动API服务
python -m voice_interaction.api.app
```

API端点:
- `POST /recognize`: 语音识别
- `POST /synthesize`: 语音合成
- `POST /assess/interview`: 面试评估
- `POST /assess/research`: 科研评估
- `POST /visualize`: 数据可视化
- `GET /health`: 健康检查

## 日志说明

日志文件保存在 `data/logs/` 目录下:

### 面试日志
- `interview_emotion_log_{timestamp}.csv`: 面试评估日志
- `interview_emotion_log_{timestamp}.json`: 面试评估JSON日志

### 科研日志
- `research_emotion_log_{timestamp}.csv`: 科研评估日志
- `research_emotion_log_{timestamp}.json`: 科研评估JSON日志

### 日志字段
- `unix_timestamp`: Unix时间戳
- `timestamp`: ISO 8601格式时间
- `pitch_mean`: 平均音高
- `pitch_variation`: 音高变化
- `pitch_trend`: 语调趋势
- `pitch_direction`: 语调方向
- `energy_mean`: 平均能量
- `energy_variation`: 能量变化
- `speech_ratio`: 语音比例
- `duration_sec`: 持续时间(秒)
- `pause_duration_mean`: 平均停顿时间(秒)
- `pause_duration_max`: 最长停顿时间(秒)
- `pause_frequency`: 停顿频率(次/分钟)
- `emotion`: 情绪状态
- `feedback`: 反馈信息
- `question_index`: 问题索引
- `is_valid`: 是否有效

## 输出说明

分析结果保存在 `data/output/voice_interaction/` 目录下:

- 热力图: `{log_filename}_heatmap.png`
- 统计图表: 面试/科研评估图表
- 分析报告: 综合评估报告

## 注意事项

1. **音频质量**: 确保音频清晰，避免噪音干扰
2. **采样率**: 使用16kHz采样率以获得最佳效果
3. **语音时长**: 单次回答建议控制在30-120秒
4. **环境噪音**: 尽量在安静环境中使用

## 性能优化

- 使用GPU加速librosa特征提取
- 降低音频采样率以提升处理速度
- 调整特征提取参数平衡准确率和速度

## 故障排除

### 问题: 语音识别不准确
- 检查音频质量
- 确认使用正确的语言模型
- 调整麦克风音量

### 问题: 特征提取失败
- 确认音频采样率为16kHz
- 检查音频文件是否损坏
- 调整音高频率范围(fmin, fmax)

### 问题: 评估结果不合理
- 确保回答内容完整
- 调整评估阈值参数
- 检查日志数据是否完整

### 问题: 可视化图表异常
- 确认日志文件格式正确
- 检查数据是否完整
- 调整可视化参数

## 版本历史

- v2.0.0: 当前版本
  - 重构模块结构
  - 优化特征提取流程
  - 改进评估算法
  - 增强可视化功能

## 许可证

Copyright © JingXin Team

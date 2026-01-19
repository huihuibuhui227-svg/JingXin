
# 手势与肢体语言分析模块 (Gesture Analysis Module)

## 模块简介

本模块是JingXin项目的子模块，提供实时手势检测、肩部动作分析和情绪评估功能，用于评估用户的抗压能力和情绪状态。

## 模块结构

```
gesture_analysis/
├── __init__.py              # 模块初始化文件
├── README.md                # 模块说明文档
├── config.py                # 配置文件
├── analyzers/               # 分析器模块
│   ├── __init__.py
│   ├── hand_analyzer.py     # 手部分析器
│   └── shoulder_analyzer.py # 肩部分析器
├── inference/               # 推理模块
│   ├── __init__.py
│   └── emotion_inferencer.py # 情绪推断器
├── api/                     # API模块
│   ├── __init__.py
│   └── app.py              # FastAPI应用
├── examples/                # 示例脚本
│   ├── __init__.py
│   └── run_realtime_analyzer.py # 实时分析示例
└── utils/                   # 工具模块
    ├── __init__.py
    ├── visualization.py     # 可视化工具
    └── logger.py           # 日志工具
```

## 功能特性

### 1. 手部分析
- 实时检测手部关键点
- 计算手指抖动幅度
- 检测握拳状态
- 计算手指张开度
- 评估抗压能力评分

### 2. 肩部分析
- 实时检测肩部位置
- 自动校准自然肩位
- 计算肩部抖动幅度
- 检测耸肩动作
- 评估紧张度评分

### 3. 情绪评估
- 综合手部和肩部特征
- 推断情绪状态（非常放松、放松、中性、轻微紧张、紧张、高度焦虑）
- 提供实时反馈和建议

## 使用方法

### 1. 基本使用

```python
from gesture_analysis import HandAnalyzer, ShoulderAnalyzer, EmotionInferencer

# 初始化分析器
left_hand_analyzer = HandAnalyzer(hand_id=0)
right_hand_analyzer = HandAnalyzer(hand_id=1)
shoulder_analyzer = ShoulderAnalyzer()
emotion_inferencer = EmotionInferencer()

# 更新手部数据
left_hand_analyzer.update(hand_landmarks)
right_hand_analyzer.update(hand_landmarks)

# 更新肩部数据
shoulder_analyzer.update(pose_landmarks)

# 推断情绪
emotion_result = emotion_inferencer.infer_emotion(
    left_hand_analyzer.get_results(),
    shoulder_analyzer.get_results()
)
```

### 2. 运行示例脚本

```bash
# 运行实时分析
python gesture_analysis/examples/run_realtime_analyzer.py
```

### 3. 使用API

```bash
# 启动API服务
python gesture_analysis/api/app.py
```

API端点：
- `GET /` - API信息
- `GET /health` - 健康检查
- `POST /analyze` - 分析图片中的手势和肩部
- `POST /reset` - 重置分析器状态

## 配置说明

在`config.py`中可以配置以下参数：

1. **MediaPipe配置**：
   - `hands` - 手部检测参数
   - `pose` - 姿态检测参数

2. **手部分析配置**：
   - `history_length` - 历史数据长度
   - `fist_threshold` - 握拳阈值
   - `jitter_multiplier` - 抖动惩罚系数
   - `spread_bonus_multiplier` - 张开度奖励系数

3. **肩部分析配置**：
   - `history_length` - 历史数据长度
   - `baseline_frames_needed` - 校准帧数
   - `jitter_multiplier` - 抖动惩罚系数
   - `shrug_penalty` - 耸肩惩罚

4. **情绪评估配置**：
   - `hand_weight` - 手部权重
   - `shoulder_weight` - 肩部权重
   - `score_ranges` - 评分范围

5. **可视化配置**：
   - `font_path` - 中文字体路径
   - `font_size` - 字体大小
   - `colors` - 颜色配置

## 依赖要求

- opencv-python>=4.5.0
- mediapipe>=0.10.0
- numpy>=1.19.0
- Pillow>=8.0.0
- fastapi>=0.110.0
- uvicorn>=0.27.0

## 注意事项

1. 首次运行时需要1-2秒校准肩部基准位置，请保持自然坐姿
2. 确保摄像头能够清晰捕获手部和肩部
3. 如需显示中文，请确保系统安装了中文字体（如simhei.ttf）
4. 建议在光线充足的环境下使用

## 许可证

本模块是JingXin项目的一部分，遵循项目的许可证协议。

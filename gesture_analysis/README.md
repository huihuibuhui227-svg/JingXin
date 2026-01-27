# Gesture Analysis Module

## 概述

手势与肢体语言分析模块，提供实时手势检测、肩部动作分析和情绪评估功能。该模块基于MediaPipe技术，能够实时分析手部、肩部、手臂和上半身的动作特征，用于评估面试者的紧张程度和情绪状态。

## 功能特性

### 核心功能
- **手部分析**: 检测手势、握拳状态、手指张开度、手部抖动
- **肩部分析**: 评估肩部稳定性、耸肩程度、抖动情况
- **手臂分析**: 分析手臂角度、稳定性、肘部抖动
- **上半身分析**: 评估头部姿态、躯干稳定性
- **情绪融合**: 综合多部位特征评估整体情绪状态

### 分析能力
- 实时手势识别和分类
- 手部动作流畅度评估
- 肩部基线自动校准
- 手臂角度和稳定性分析
- 头部姿态和偏转检测
- 躯干稳定性评估
- 多模态情绪融合

## 模块结构

```
gesture_analysis/
├── __init__.py                 # 模块初始化
├── config.py                   # 配置文件
├── api/
│   ├── __init__.py
│   └── app.py                # FastAPI接口
├── core/
│   ├── __init__.py
│   ├── analysis/              # 分析器
│   │   ├── hand_analyzer.py
│   │   ├── shoulder_analyzer.py
│   │   ├── arm_analyzer.py
│   │   ├── upper_body_analyzer.py
│   │   └── emotion_inferencer.py
│   └── feature_extraction/    # 特征提取
├── models/
│   ├── __init__.py
│   └── data_models.py        # 数据模型
├── pipeline/
│   ├── __init__.py
│   └── gesture_pipeline.py   # 手势分析流程
├── utils/
│   ├── __init__.py
│   ├── logger.py            # 日志工具
│   └── visualization.py     # 可视化工具
└── examples/
    ├── __init__.py
    ├── run_realtime_analyzer.py
    └── visualize_logs.py
```

## 安装依赖

```bash
pip install mediapipe opencv-python numpy pandas matplotlib pillow
```

## 配置说明

### 路径配置
- `DATA_DIR`: 数据根目录 (默认: `data`)
- `INPUT_DIR`: 输入文件目录 (默认: `data/input`)
- `OUTPUT_DIR`: 输出结果目录 (默认: `data/output`)
- `LOGS_DIR`: 日志文件目录 (默认: `data/logs`)
- `GESTURE_OUTPUT_DIR`: 手势分析输出目录 (默认: `data/output/gesture_analysis`)

### MediaPipe配置
```python
MEDIAPIPE_CONFIG = {
    'hands': {
        'static_image_mode': False,
        'max_num_hands': 2,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.5
    },
    'pose': {
        'static_image_mode': False,
        'model_complexity': 1,
        'min_detection_confidence': 0.6,
        'min_tracking_confidence': 0.6
    }
}
```

### 手部分析配置
```python
HAND_CONFIG = {
    'history_length': 30,              # 历史数据长度(帧)
    'fist_threshold': 0.08,            # 握拳判定阈值
    'jitter_multiplier': 1000.0,        # 抖动惩罚系数
    'spread_bonus_multiplier': 80.0,      # 手指张开奖励系数
    'spread_threshold': 0.25,           # 张开度阈值
    'fist_penalty': 20.0                # 握拳惩罚分
}
```

### 手臂分析配置
```python
ARM_CONFIG = {
    'history_length': 30,              # 历史数据长度(帧)
    'jitter_multiplier': 1500.0,        # 抖动惩罚系数
    'ideal_angle_min': 70.0,           # 理想手臂角度最小值(度)
    'ideal_angle_max': 110.0,          # 理想手臂角度最大值(度)
    'acceptable_angle_min': 50.0,       # 可接受手臂角度最小值(度)
    'acceptable_angle_max': 130.0,      # 可接受手臂角度最大值(度)
    'stability_bonus': 20.0             # 稳定性奖励系数
}
```

### 肩部分析配置
```python
SHOULDER_CONFIG = {
    'history_length': 30,               # 历史数据长度(帧)
    'baseline_frames_needed': 30,         # 初始校准所需帧数
    'jitter_multiplier': 2000.0,        # 肩部抖动惩罚系数
    'shrug_penalty': 30.0,             # 耸肩惩罚分
    'max_shrug_diff': 0.1,             # 左右肩高度差阈值
    'baseline_smoothing': 0.9            # 基准线平滑因子
}
```

### 情绪评估配置
```python
EMOTION_CONFIG = {
    'hand_weight': 0.4,                 # 手部特征权重
    'shoulder_weight': 0.3,             # 肩部特征权重
    'arm_weight': 0.3,                 # 手臂特征权重
    'score_ranges': {
        'very_relaxed': 80,             # 非常放松阈值
        'relaxed': 65,                 # 放松阈值
        'neutral': 50,                  # 中性阈值
        'slightly_nervous': 35,         # 轻微紧张阈值
        'nervous': 20                  # 紧张阈值
    }
}
```

## 使用方法

### 实时手势分析

```python
from gesture_analysis import GestureEmotionPipeline
import cv2

# 初始化管道
pipeline = GestureEmotionPipeline()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 分析帧
    results = pipeline.process_frame(frame)

    # 获取情绪结果
    emotion_result = pipeline.get_emotion_result()
    print(f"情绪: {emotion_result['emotion_state']}")
    print(f"评分: {emotion_result['overall_score']}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 使用分析器

```python
from gesture_analysis import HandAnalyzer, ShoulderAnalyzer, EmotionFusionAnalyzer

# 初始化分析器
hand_analyzer = HandAnalyzer(hand_id=0)
shoulder_analyzer = ShoulderAnalyzer()
emotion_analyzer = EmotionFusionAnalyzer()

# 更新和分析
hand_analyzer.update(landmarks)
hand_result = hand_analyzer.get_results()

shoulder_analyzer.update(pose_landmarks)
shoulder_result = shoulder_analyzer.get_results()

# 融合情绪
emotion_result = emotion_analyzer.fuse_emotion(
    hand_result=hand_result,
    shoulder_result=shoulder_result
)
```

### 使用API服务

```bash
# 启动API服务
python -m gesture_analysis.api.app
```

API端点:
- `POST /analyze/frame`: 分析单帧图像
- `POST /analyze/video`: 分析视频流
- `GET /health`: 健康检查

## 日志说明

日志文件保存在 `data/logs/` 目录下:

- `gesture_emotion_log_{timestamp}.csv`: 手势分析日志

日志字段:
- **时间戳**: `timestamp`, `timestamp_iso`
- **手部特征**: `left_hand_score`, `right_hand_score`, `left_hand_jitter`, `right_hand_jitter`, `left_hand_fist_status`, `right_hand_fist_status`, `left_hand_spread`, `right_hand_spread`
- **手指角度**: `left_thumb_angle`, `left_index_angle`, `left_middle_angle`, `left_ring_angle`, `left_pinky_angle`, `right_thumb_angle`, `right_index_angle`, `right_middle_angle`, `right_ring_angle`, `right_pinky_angle`
- **肩部特征**: `shoulder_score`, `left_shoulder_jitter`, `right_shoulder_jitter`, `shrug_level`, `is_calibrated`
- **手臂特征**: `left_arm_score`, `right_arm_score`, `left_wrist_jitter`, `right_wrist_jitter`, `left_elbow_jitter`, `right_elbow_jitter`, `left_arm_angle`, `right_arm_angle`, `left_arm_stability`, `right_arm_stability`
- **手臂角度**: `left_elbow_angle`, `right_elbow_angle`, `left_shoulder_angle`, `right_shoulder_angle`
- **上半身特征**: `head_score`, `head_jitter`, `head_tilt`, `torso_score`, `torso_jitter`, `torso_stability`
- **头部角度**: `head_tilt_angle`, `head_pitch_angle`
- **肩部角度**: `shoulder_angle`
- **躯干角度**: `torso_angle`
- **情绪特征**: `overall_score`, `emotion_state`, `feedback`, `used_features`, `is_valid`

## 输出说明

分析结果保存在 `data/output/gesture_analysis/` 目录下:

- 可视化视频/图片
- 分析报告
- 统计图表

## 注意事项

1. **摄像头位置**: 确保摄像头正对上半身，包含手部
2. **光照条件**: 确保光线充足均匀
3. **距离**: 建议距离摄像头1-2米
4. **校准**: 启动后保持自然坐姿30秒以完成肩部校准
5. **双手可见**: 确保双手在摄像头视野内

## 性能优化

- 使用GPU加速MediaPipe推理
- 降低视频分辨率以提升处理速度
- 调整检测置信度阈值平衡准确率和速度
- 减少历史数据长度以降低内存使用

## 故障排除

### 问题: 检测不到手部
- 检查双手是否在摄像头视野内
- 确保光照充足
- 调整 `min_detection_confidence` 参数

### 问题: 肩部校准失败
- 保持自然坐姿30秒
- 确保上半身完整在视野内
- 避免大幅度动作干扰校准

### 问题: 情绪评估不准确
- 确保肩部已校准
- 调整 `EMOTION_CONFIG` 中的权重和阈值
- 检查各部位分析结果是否有效

### 问题: 性能较慢
- 降低视频分辨率
- 调整 `max_num_hands` 参数
- 启用GPU加速

## 版本历史

- v1.0.0: 初始版本
  - 实现基础手势检测
  - 添加肩部和手臂分析
  - 实现多模态情绪融合
  - 支持实时分析

## 许可证

Copyright © JingXin Team

# Face Expression 模块依赖关系分析

## 一、模块结构

```
face_expression/
├── __init__.py              # 模块初始化，导出主要接口
├── config.py                # 配置文件，无外部依赖
├── analyzers/               # 分析器模块
│   ├── __init__.py
│   ├── landmarks.py         # 面部关键点索引配置，无外部依赖
│   ├── face_au_analyzer.py  # 实时视频分析器，依赖MediaPipe
│   └── image_analyzer.py    # 静态图片分析器，依赖MediaPipe
├── inference/               # 推理模块
│   ├── __init__.py
│   └── emotion_infer.py     # 情绪推断引擎，无外部依赖
├── api/                     # API模块
│   ├── __init__.py
│   └── app.py              # FastAPI应用，依赖StaticFaceAnalyzer
├── examples/                # 示例脚本
│   ├── __init__.py
│   ├── run_video_analyzer.py # 实时视频分析示例，依赖FaceAUAnalyzer
│   ├── run_image_analyzer.py # 静态图片分析示例，依赖StaticFaceAnalyzer
│   └── visualize_logs.py    # 日志可视化示例，依赖plot_features_from_csv
├── tests/                   # 测试模块
│   ├── __init__.py
│   └── test_face_expression.py
└── utils/                   # 工具模块
    ├── __init__.py
    ├── logger.py           # 日志工具，依赖config.py
    └── visualize.py        # 数据可视化工具，无外部依赖
```

## 二、模块依赖关系

### 1. 核心依赖链

```
config.py (无依赖)
    ↓
logger.py (依赖config.py)
    ↓
emotion_infer.py (无依赖)
    ↓
landmarks.py (无依赖)
    ↓
face_au_analyzer.py (依赖landmarks.py, emotion_infer.py, MediaPipe)
    ↓
image_analyzer.py (依赖landmarks.py, emotion_infer.py, MediaPipe)
    ↓
StaticFaceAnalyzer (来自image_analyzer.py)
    ↓
FaceAUAnalyzer (来自face_au_analyzer.py)
```

### 2. 模块间调用关系

#### analyzers模块
- **landmarks.py**: 无依赖，独立模块
- **face_au_analyzer.py**: 
  - 依赖: landmarks.py, emotion_infer.py, MediaPipe
  - 被依赖: __init__.py, run_video_analyzer.py
- **image_analyzer.py**:
  - 依赖: landmarks.py, emotion_infer.py, MediaPipe
  - 被依赖: __init__.py, run_image_analyzer.py, api/app.py

#### inference模块
- **emotion_infer.py**: 
  - 依赖: 无
  - 被依赖: face_au_analyzer.py, image_analyzer.py

#### utils模块
- **logger.py**:
  - 依赖: config.py
  - 被依赖: __init__.py
- **visualize.py**:
  - 依赖: 无
  - 被依赖: __init__.py, visualize_logs.py

#### examples模块
- **run_video_analyzer.py**:
  - 依赖: FaceAUAnalyzer, MediaPipe绘图工具
- **run_image_analyzer.py**:
  - 依赖: StaticFaceAnalyzer
- **visualize_logs.py**:
  - 依赖: plot_features_from_csv

## 三、可独立运行的模块

### 1. 完全独立（无外部依赖）

以下模块可以独立运行，不依赖其他模块或外部库：

1. **landmarks.py**
   - 功能: 定义面部关键点索引
   - 依赖: 无
   - 可直接导入使用

2. **emotion_infer.py**
   - 功能: 基于AU特征推断情绪
   - 依赖: 无
   - 可直接导入使用

3. **visualize.py**
   - 功能: 可视化日志数据
   - 依赖: pandas, matplotlib
   - 可直接运行: `python face_expression/utils/visualize.py`

### 2. 依赖配置模块

1. **logger.py**
   - 功能: 数据日志记录
   - 依赖: config.py
   - 可独立运行，只需确保config.py存在

### 3. 依赖MediaPipe（需要正确安装MediaPipe）

1. **face_au_analyzer.py**
   - 功能: 实时视频流AU分析
   - 依赖: landmarks.py, emotion_infer.py, MediaPipe
   - 需要正确安装和导入MediaPipe

2. **image_analyzer.py**
   - 功能: 静态图片AU分析
   - 依赖: landmarks.py, emotion_infer.py, MediaPipe
   - 需要正确安装和导入MediaPipe

### 4. 示例脚本

1. **run_image_analyzer.py**
   - 功能: 静态图片分析示例
   - 依赖: StaticFaceAnalyzer
   - 需要MediaPipe正确工作

2. **run_video_analyzer.py**
   - 功能: 实时视频分析示例
   - 依赖: FaceAUAnalyzer, MediaPipe绘图工具
   - 需要MediaPipe正确工作

3. **visualize_logs.py**
   - 功能: 日志可视化示例
   - 依赖: plot_features_from_csv
   - 需要先有日志数据

## 四、当前问题分析

### MediaPipe导入问题

当前主要问题是MediaPipe的导入失败：
```python
mp_face_mesh = mp.solutions.face_mesh
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

这可能的原因：
1. MediaPipe版本不兼容
2. MediaPipe未正确安装
3. 导入方式不正确

### 解决方案

#### 方案1: 重新安装MediaPipe

```bash
# 卸载当前版本
pip uninstall mediapipe

# 安装指定版本
pip install mediapipe==0.10.21
```

#### 方案2: 检查MediaPipe版本

```python
import mediapipe as mp
print(mp.__version__)
```

#### 方案3: 使用dlib替代（需要重写代码）

如果MediaPipe问题无法解决，可以考虑使用dlib重写面部特征点检测部分。

## 五、建议的测试顺序

1. 首先测试无依赖的模块：
   ```python
   from face_expression.inference import infer_emotion_from_au
   features = {
       'au4_frown': 0.2,
       'au12_eyebrow_raise': 0.05,
       'au12_smile': 0.7,
       'au9_nose_wrinkle': 0.1,
       'au15_mouth_down': 0.02,
       'au25_mouth_open': 0.1,
       'focus_score': 0.7
   }
   emotion = infer_emotion_from_au(features)
   print(emotion)
   ```

2. 测试landmarks模块：
   ```python
   from face_expression.analyzers import LEFT_EYE, RIGHT_EYE
   print(LEFT_EYE)
   print(RIGHT_EYE)
   ```

3. 解决MediaPipe问题后再测试分析器

4. 最后测试完整的示例脚本

## 六、模块调用关系图

```
run_video_analyzer.py
    ├── FaceAUAnalyzer
    │   ├── landmarks.py
    │   ├── emotion_infer.py
    │   └── MediaPipe
    └── MediaPipe绘图工具

run_image_analyzer.py
    └── StaticFaceAnalyzer
        ├── landmarks.py
        ├── emotion_infer.py
        └── MediaPipe

visualize_logs.py
    └── plot_features_from_csv
        └── pandas, matplotlib

api/app.py
    └── StaticFaceAnalyzer
        ├── landmarks.py
        ├── emotion_infer.py
        └── MediaPipe
```

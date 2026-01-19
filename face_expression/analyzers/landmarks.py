"""
面部关键点索引配置

基于 MediaPipe FaceMesh 的 468 个关键点定义常用区域索引。
所有索引均为常量，不可修改。
"""

from typing import List

# 眼睛区域
LEFT_EYE: List[int] = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE: List[int] = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_UPPER: List[int] = [159, 158, 157]
RIGHT_EYE_UPPER: List[int] = [386, 385, 384]

# 眉毛区域
LEFT_EYEBROW_UPPER: List[int] = [107, 105, 104, 103, 102]
RIGHT_EYEBROW_UPPER: List[int] = [336, 334, 333, 332, 331]
BROW_CENTER_LEFT: int = 66
BROW_CENTER_RIGHT: int = 296

# 鼻子区域
NOSE_TIP: int = 1
NOSE_WING_LEFT: int = 129
NOSE_WING_RIGHT: int = 358

# 嘴巴区域
MOUTH_CORNER_LEFT: int = 61
MOUTH_CORNER_RIGHT: int = 291
MOUTH_CENTER_TOP: int = 13
MOUTH_CENTER_BOTTOM: int = 14
LIP_TOP: int = 0
LIP_BOTTOM: int = 17

# 脸部轮廓
CHIN: int = 152
EYE_CORNER_LEFT: int = 130
EYE_CORNER_RIGHT: int = 359
CHEEK_LEFT: int = 205
CHEEK_RIGHT: int = 425
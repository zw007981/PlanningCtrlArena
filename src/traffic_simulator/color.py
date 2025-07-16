# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


from PySide6.QtGui import QColor

TRANSPARENT = QColor(0, 0, 0, 0)
"""透明"""
BLACK = QColor(0, 0, 0, 255)
"""黑色"""
GREY = QColor(0, 0, 0, 150)
"""灰色"""
SEMI_TRN_RED = QColor(255, 0, 0, 127)
"""半透明红色"""
SEMI_TRN_YELLOW = QColor(255, 165, 0, 127)
"""半透明黄色"""
SEMI_TRN_BLUE = QColor(0, 0, 255, 127)
"""半透明蓝色"""
SEMI_TRN_GREEN = QColor(0, 255, 0, 127)
"""半透明绿色"""
SEMI_TRN_CYAN = QColor(0, 255, 255, 127)
"""半透明青色"""
RED = QColor(255, 0, 0, 255)
"""红色"""
YELLOW = QColor(255, 165, 0, 255)
"""黄色"""
BLUE = QColor(0, 0, 255, 255)
"""蓝色"""
GREEN = QColor(0, 255, 0, 255)
"""绿色"""
CYAN = QColor(0, 255, 255, 255)
"""青色"""

# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from src.map.obstacle import Obstacle, ObstacleType
from src.traffic_simulator.color import *
from src.util.pose import Pose


class GraphicItem(QtWidgets.QGraphicsRectItem):
    """图像数据结构基类"""

    def __init__(self, id: str, type: str = "", length=0.4, width=0.2) -> None:
        """使用设备ID和设备类型初始化设备图像数据结构"""
        self.id_ = id
        """设备ID"""
        self.type_ = type
        """设备类型"""
        self.is_visible_ = False
        """设备是否可见"""
        self.half_len_: float = length / 2
        """设备长度的一半"""
        self.half_width_: float = width / 2
        """设备宽度的一半"""
        super().__init__(QtCore.QRectF(0, 0, 2 * self.half_len_, 2 * self.half_width_))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.leftButtonPressedImpl()
        super().mousePressEvent(event)

    def leftButtonPressedImpl(self) -> None:
        """处理鼠标左键按下事件"""
        return

    def getLeftBottomPt(
        self, pose: Pose, sin: float, cos: float
    ) -> Tuple[float, float]:
        """
        根据设备中心点位姿和尺寸的一半计算设备左下角点的坐标，
        注意这里我们的中心点是后轴的中心
        """
        return (
            pose.x + self.half_width_ * sin,
            pose.y - self.half_width_ * cos,
        )
        # return (
        #     pose.x - self.half_len_ * cos + self.half_width_ * sin,
        #     pose.y - self.half_len_ * sin - self.half_width_ * cos,
        # )

    def getMiddleHeadPt(
        self, pose: Pose, sin: float, cos: float
    ) -> Tuple[float, float]:
        """根据设备中心点位姿和尺寸的一半计算设备头部中点的坐标"""
        return (
            pose.x + self.half_len_ * cos,
            pose.y + self.half_len_ * sin,
        )


class CarGraphicItem(GraphicItem):
    """汽车图像数据结构"""

    import pyqtgraph as pg

    def __init__(self, id: str, length=0.4, width=0.2, color=TRANSPARENT) -> None:
        """使用汽车ID初始化汽车图像数据结构"""
        import pyqtgraph as pg

        super().__init__(id, "car", length, width)
        self.length_ = length
        """汽车长度(此处为轴距)"""
        self.width_ = width
        """汽车宽度"""
        self.pose_ = Pose(0, 0, 0)
        """汽车位姿"""
        super().setVisible(False)
        self.setPen(pg.mkPen(BLACK, width=2))  # type: ignore
        self.setBrush(QtGui.QBrush(color))
        self.center_item_ = pg.ScatterPlotItem(pen=None, brush=YELLOW)
        """汽车中心点（后轮轴中心）"""
        self.center_item_.setSize(6)
        self.center_item_.setVisible(False)

    def addItem(self, scene: pg.PlotWidget) -> None:
        """将汽车图像添加到场景中"""
        scene.addItem(self)
        scene.addItem(self.center_item_)

    def removeItem(self, scene: pg.PlotWidget) -> None:
        """将汽车图像从场景中移除"""
        scene.removeItem(self)
        scene.removeItem(self.center_item_)

    def setColor(self, color: QtGui.QColor) -> None:
        """设置汽车颜色"""
        self.setBrush(QtGui.QBrush(color))

    def setVisibility(self, is_visible: bool) -> None:
        """设置汽车可见性"""
        super().setVisible(is_visible)
        self.center_item_.setVisible(is_visible)

    def update(self, pose: Pose) -> None:
        """更新汽车图像"""
        self.pose_ = pose
        sin = np.sin(pose.theta)
        cos = np.cos(pose.theta)
        left_bottom_x, left_bottom_y = self.getLeftBottomPt(pose, sin, cos)
        super().setPos(left_bottom_x, left_bottom_y)
        super().setRotation(np.degrees(pose.theta))
        self.center_item_.setData([pose.x], [pose.y])


class ObstacleGraphicItem:
    """障碍物图像数据结构"""

    import pyqtgraph as pg

    def __init__(self, obs_info: Obstacle):
        """使用障碍物信息初始化障碍物图像数据结构"""
        import pyqtgraph as pg

        self.obs_info = obs_info
        """障碍物信息"""
        self.graphic_item = QtWidgets.QGraphicsEllipseItem(
            0, 0, self.obs_info.radius * 2, self.obs_info.radius * 2
        )
        """障碍物图像"""
        self.graphic_item.setPen(pg.mkPen(BLACK, width=2))  # type: ignore
        if obs_info.obstacle_type == ObstacleType.STATIC:
            self.graphic_item.setBrush(QtGui.QBrush(GREY))
        else:
            self.graphic_item.setBrush(QtGui.QBrush(TRANSPARENT))
        self.graphic_item.setVisible(True)
        self.graphic_item.setPos(
            self.obs_info.x - self.obs_info.radius,
            self.obs_info.y - self.obs_info.radius,
        )
        self.observation_range_item = QtWidgets.QGraphicsEllipseItem(
            0, 0, self.obs_info.obs_threshold * 2, self.obs_info.obs_threshold * 2
        )
        """观测范围图像，一旦车辆进入该范围动态障碍物就可以被精确观察到"""
        if obs_info.obstacle_type == ObstacleType.UNCERTAIN:
            self.graphic_item.setVisible(True)
            pen = QtGui.QPen(GREY, 0.04)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            self.observation_range_item.setPen(pen)
            self.observation_range_item.setBrush(QtGui.QBrush(TRANSPARENT))
            self.observation_range_item.setPos(
                self.obs_info.x - self.obs_info.obs_threshold,
                self.obs_info.y - self.obs_info.obs_threshold,
            )

    def addItem(self, scene: pg.PlotWidget) -> None:
        """将障碍物图像添加到场景中"""
        scene.addItem(self.graphic_item)
        # if self.obs_info.obstacle_type == ObstacleType.UNCERTAIN:
        #     scene.addItem(self.observation_range_item)

    def removeItem(self, scene: pg.PlotWidget) -> None:
        """将障碍物图像从场景中移除"""
        scene.removeItem(self.graphic_item)

    def update(self) -> None:
        """更新障碍物图像"""
        if self.obs_info.obstacle_type == ObstacleType.STATIC:
            self.graphic_item.setBrush(QtGui.QBrush(GREY))
        elif self.obs_info.has_been_observed and self.obs_info.will_appear:
            self.graphic_item.setBrush(QtGui.QBrush(GREY))
        else:
            self.graphic_item.setBrush(QtGui.QBrush(TRANSPARENT))

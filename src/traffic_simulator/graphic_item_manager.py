# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import Dict, List

from numpy.typing import NDArray
from pyqtgraph import PlotWidget
from PySide6 import QtCore
from PySide6.QtGui import QColor

from src.algo.level_k.level import Level
from src.traffic_simulator.color import *
from src.traffic_simulator.data_manager import DataManager
from src.traffic_simulator.graphic_item import CarGraphicItem, ObstacleGraphicItem
from src.util.config import Config
from src.util.log import Log


class GraphicItemManager:
    """图形项管理器，用于统一管理PySide6中的图形项"""

    def __init__(self, data_mgr: DataManager, graph_widget: PlotWidget) -> None:
        """基于数据管理器中的数据初始化图形项管理器"""
        import pyqtgraph as pg

        self.graph_widget = graph_widget
        """绘图窗口"""
        self.car_id_to_graphic_item: Dict[str, CarGraphicItem] = {}
        """从车辆ID到车辆图形项的映射"""
        self.__initCarGraphicItems(data_mgr)
        self.ref_pt_item = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush("r")
        )
        """参考点图形项"""
        self.graph_widget.addItem(self.ref_pt_item)
        self.ref_line_item = pg.PlotCurveItem(
            name="reference line",
            connect="pairs",
            pen=pg.mkPen(color=SEMI_TRN_GREEN, style=QtCore.Qt.DashLine, width=4),  # type: ignore
        )
        """参考线图形项"""
        self.__initRefLineItem(data_mgr)
        self.map_bound_item = pg.PlotCurveItem(
            connect="pairs",
            pen=pg.mkPen(color=BLACK, style=QtCore.Qt.SolidLine, width=8),  # type: ignore
        )
        """地图边界线图形项"""
        self.__initMapBoundItem(data_mgr)
        self.obstacle_items: List[ObstacleGraphicItem] = []
        """障碍物图形项列表"""
        self.__initObstacleItems(data_mgr)
        self.ego_car_trj_item = pg.PlotCurveItem(
            name="trajectory",
            pen=pg.mkPen(color=SEMI_TRN_YELLOW, style=QtCore.Qt.SolidLine, width=3),  # type: ignore
        )
        """主车轨迹图形项"""
        self.ego_car_trj_item.setData([], [])
        self.graph_widget.addItem(self.ego_car_trj_item)
        self.extra_curve_items: Dict[str, pg.PlotCurveItem] = {}
        """在某些情况下需要额外绘制的曲线图形项"""

    def __initCarGraphicItems(self, data_mgr: DataManager) -> None:
        """初始化车辆图形项"""
        for car_id, car in data_mgr.equip_mgr.id_to_status.items():
            self.car_id_to_graphic_item[car_id] = CarGraphicItem(
                car_id, Config.WHEEL_BASE, Config.WIDTH, TRANSPARENT
            )
            if data_mgr.ego_car.id == car_id:
                # 主车为青色
                self.car_id_to_graphic_item[car_id].setColor(SEMI_TRN_CYAN)
            elif car.level == Level.ADAPTIVE:
                # 自适应级别的车为灰色
                self.car_id_to_graphic_item[car_id].setColor(GREY)
            elif car.level == Level.LEVEL_0 or car.level == Level.LEVEL_2:
                # 激进的偶数级别为黄色
                self.car_id_to_graphic_item[car_id].setColor(SEMI_TRN_YELLOW)
            elif car.level == Level.LEVEL_1:
                # 保守的奇数级别为透明矩形
                self.car_id_to_graphic_item[car_id].setColor(TRANSPARENT)
            self.car_id_to_graphic_item[car_id].addItem(self.graph_widget)
            self.car_id_to_graphic_item[car_id].setVisibility(True)
            self.car_id_to_graphic_item[car_id].update(car.pose)

    def __initRefLineItem(self, data_mgr: DataManager) -> None:
        """初始化参考线图形项"""
        x_list, y_list = [], []
        for ref_line in data_mgr.map.ref_lines:
            for i in range(0, len(ref_line.pts) - 1, 3):
                pose0 = ref_line.pts[i]
                pose1 = ref_line.pts[i + 1]
                x_list.append(pose0.x)
                x_list.append(pose1.x)
                y_list.append(pose0.y)
                y_list.append(pose1.y)
        self.ref_line_item.setData(x_list, y_list)
        self.graph_widget.addItem(self.ref_line_item)

    def __initMapBoundItem(self, data_mgr: DataManager) -> None:
        """初始化地图边界线图形项"""
        x_list = [
            Config.X_MIN,
            Config.X_MAX,
            Config.X_MAX,
            Config.X_MAX,
            Config.X_MAX,
            Config.X_MIN,
            Config.X_MIN,
            Config.X_MIN,
        ]
        y_list = [
            Config.Y_MIN,
            Config.Y_MIN,
            Config.Y_MIN,
            Config.Y_MAX,
            Config.Y_MAX,
            Config.Y_MAX,
            Config.Y_MAX,
            Config.Y_MIN,
        ]
        self.map_bound_item.setData(x_list, y_list)
        self.graph_widget.addItem(self.map_bound_item)

    def __initObstacleItems(self, data_mgr: DataManager) -> None:
        """初始化障碍物图形项"""
        for obstacle in data_mgr.map.obstacles:
            self.obstacle_items.append(ObstacleGraphicItem(obstacle))
            self.obstacle_items[-1].addItem(self.graph_widget)

    def update(self, data_mgr: DataManager) -> None:
        """更新图形项"""
        for obstacle in self.obstacle_items:
            obstacle.update()
        for car_id, car in data_mgr.equip_mgr.id_to_status.items():
            self.car_id_to_graphic_item[car_id].update(car.pose)
        self.ref_pt_item.setData(
            [data_mgr.id_to_ref_pt[car_id].x for car_id in data_mgr.car_id_list],
            [data_mgr.id_to_ref_pt[car_id].y for car_id in data_mgr.car_id_list],
        )
        if len(data_mgr.ego_car.trajectory) >= 2:
            delta_x = abs(
                data_mgr.ego_car.trajectory[-1].x - data_mgr.ego_car.trajectory[0].x
            )
            # 不知道是因为版本的问题还是因为刚起步的时候轨迹过短，
            # 不加这个条件起始的时候轨迹线会跳一下。
            if delta_x > Config.ZERO_EPS:
                self.ego_car_trj_item.setData(
                    [pose.x for pose in data_mgr.ego_car.trajectory],
                    [pose.y for pose in data_mgr.ego_car.trajectory],
                )

    def reset(self, data_mgr: DataManager) -> None:
        """重置图形项"""
        for obstacle in self.obstacle_items:
            obstacle.update()
        for car_id, car in data_mgr.equip_mgr.id_to_status.items():
            self.car_id_to_graphic_item[car_id].setVisibility(True)
            self.car_id_to_graphic_item[car_id].update(car.pose)
        self.ref_pt_item.setData(
            [data_mgr.id_to_ref_pt[car_id].x for car_id in data_mgr.car_id_list],
            [data_mgr.id_to_ref_pt[car_id].y for car_id in data_mgr.car_id_list],
        )
        self.ego_car_trj_item.setData([], [])
        for _, extra_curve_item in self.extra_curve_items.items():
            extra_curve_item.setData([], [])

    def addExtraCurveItem(
        self, name: str, color: QColor, style=QtCore.Qt.PenStyle.SolidLine, width=3
    ) -> None:
        """添加额外曲线图形项"""
        import pyqtgraph as pg

        if name in self.extra_curve_items.keys():
            Log.warning("Extra curve item {} already exists!".format(name))
            return
        self.extra_curve_items[name] = pg.PlotCurveItem(
            name=name, pen=pg.mkPen(color=color, style=style, width=width)
        )
        self.graph_widget.addItem(self.extra_curve_items[name])

    def setExtraCurveData(self, name: str, data: NDArray) -> None:
        """设置额外曲线图形项数据"""
        if name not in self.extra_curve_items.keys():
            Log.warning("Extra curve item {} not exists!".format(name))
            return
        self.extra_curve_items[name].setData(data[:, 0], data[:, 1])

    def getCarGraphicItem(self, car_id: str) -> CarGraphicItem:
        """获取车辆图形项"""
        if car_id not in self.car_id_to_graphic_item.keys():
            Log.error("Car {} not in graphic item manager!!!".format(car_id))
            raise ValueError("Car not in graphic item manager!!!")
        return self.car_id_to_graphic_item[car_id]

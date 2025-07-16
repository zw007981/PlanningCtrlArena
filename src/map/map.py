# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import json
import math
from typing import List, Tuple

from src.map.obstacle import Obstacle
from src.map.point_finder import PointFinder
from src.map.reference_line import ReferenceLine
from src.util.config import Config
from src.util.log import Log
from src.util.pose import Pose


class Map:
    def __init__(self):
        self.ref_lines: List[ReferenceLine] = []
        """参考线列表"""
        self.ref_line_poses: List[Pose] = []
        """参考线上的所有点"""
        self.pt_finder = PointFinder()
        """点位查找器，用于快速查找最近的参考线点"""
        self.obstacles: List[Obstacle] = []
        """障碍物列表"""
        self.x_min = math.inf
        """地图的最小x坐标"""
        self.x_max = -math.inf
        """地图的最大x坐标"""
        self.y_min = math.inf
        """地图的最小y坐标"""
        self.y_max = -math.inf
        """地图的最大y坐标"""

    def getRefLine(self, ref_line_id: int) -> ReferenceLine:
        """获取参考线"""
        if ref_line_id >= 0 and ref_line_id < len(self.ref_lines):
            return self.ref_lines[ref_line_id]
        else:
            Log.error("Invalid reference line id: %d!!!" % ref_line_id)
            raise ValueError("Invalid reference line id: %d!!!" % ref_line_id)

    def getNearestRefPt(self, x: float, y: float, ref_line_id: int = -1) -> Pose:
        """获取最近的参考线点，如果指定了ref_line_id，则只在该参考线上查找"""
        if ref_line_id >= 0 and ref_line_id < len(self.ref_lines):
            ref_line = self.ref_lines[ref_line_id]
            return ref_line.getNearestPt(x, y)
        else:
            nearest_ref_pt_index = self.pt_finder.findIndex(x, y)
            return self.ref_line_poses[nearest_ref_pt_index]

    def getNearestRefPtInRng(
        self,
        x: float,
        y: float,
        ref_line_id: int,
        start_index: int,
        stop_index: int = -1,
    ) -> Pose:
        """获取指定参考线上编号在[start_index, stop_index)范围内的最近的参考线点"""
        return self.ref_lines[ref_line_id].getNearestPtInRng(
            x, y, start_index, stop_index
        )

    def getNearestRefPtInDist(
        self, x: float, y: float, ref_line_id: int, start_index: int, dist: float
    ) -> Pose:
        """获取指定参考线上从start_index开始，距离[x,y]距离不超过dist的最近参考点"""
        return self.ref_lines[ref_line_id].getNearestPtInDist(x, y, start_index, dist)

    def getNearestRefPtAfterDistance(
        self, ref_line_id: int, start_index: int, dist: float
    ) -> Pose:
        """获取指定参考线上编号为start_index的点之后距离为dist的参考线点"""
        return self.ref_lines[ref_line_id].getPtAfterDistance(start_index, dist)

    def loadMapFromFile(self, file_name: str):
        """
        从config文件夹里寻找是否有名字为file_name的json文件，有的话就打开并读取那份文件
        """
        self.__clear()
        # 读取控制点信息
        file_path = os.path.join(Config.GetConfigFolder(), file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                for ref_line in data["reference_lines"]:
                    ctrl_pts: List[Tuple[float, float]] = []
                    for pt in ref_line["control_points"]:
                        ctrl_pts.append((float(pt["x"]), float(pt["y"])))
                    self.ref_lines.append(ReferenceLine(ctrl_pts))
                if "obstacles" in data:
                    for obs_data in data["obstacles"]:
                        self.obstacles.append(Obstacle(obs_data))
        else:
            raise FileNotFoundError(f"File {file_name} not found in path {file_path}")
        for ref_line in self.ref_lines:
            self.x_min = min(self.x_min, ref_line.x_min)
            self.x_max = max(self.x_max, ref_line.x_max)
            self.y_min = min(self.y_min, ref_line.y_min)
            self.y_max = max(self.y_max, ref_line.y_max)
            for pt in ref_line.pts:
                self.ref_line_poses.append(pt)
                self.pt_finder.addPt(pt.x, pt.y)
        self.pt_finder.buildKDTree()
        self.__updateMapSize()

    def __clear(self):
        """清空地图数据"""
        self.ref_lines = []
        self.ref_line_poses = []
        self.pt_finder = PointFinder()
        self.obstacles = []
        self.x_min = math.inf
        self.x_max = -math.inf
        self.y_min = math.inf
        self.y_max = -math.inf

    def __updateMapSize(self):
        """更新地图尺寸信息"""
        expandIfTooSmall = lambda min_val, max_val: (
            (min_val - 3.0, max_val + 3.0)
            if max_val - min_val < 6.0
            else (min_val, max_val)
        )
        self.x_min = math.floor(self.x_min / 2.0) * 2.0 - 2.0
        self.x_max = math.ceil(self.x_max / 2.0) * 2.0 + 2.0
        self.y_min = math.floor(self.y_min / 2.0) * 2.0 - 2.0
        self.y_max = math.ceil(self.y_max / 2.0) * 2.0 + 2.0
        self.x_min, self.x_max = expandIfTooSmall(self.x_min, self.x_max)
        self.y_min, self.y_max = expandIfTooSmall(self.y_min, self.y_max)
        Config.X_MIN = self.x_min
        Config.X_MAX = self.x_max
        Config.Y_MIN = self.y_min
        Config.Y_MAX = self.y_max


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    map = Map()
    map.loadMapFromFile("config.json")
    for ref_line in map.ref_lines:
        for i in range(len(ref_line.pts) - 1):
            plt.plot(
                [ref_line.pts[i].x, ref_line.pts[i + 1].x],
                [ref_line.pts[i].y, ref_line.pts[i + 1].y],
                "r-",
            )
        plt.plot(
            ref_line.ctrl_pts[:, 0],
            ref_line.ctrl_pts[:, 1],
            "bo",
            label="control points",
        )
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

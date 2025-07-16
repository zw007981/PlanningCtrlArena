# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import math
from typing import List, Tuple

import numpy as np

from src.map.point_finder import PointFinder
from src.util.B_spline import BSpline
from src.util.config import Config
from src.util.curvature import Curvature
from src.util.log import Log
from src.util.pose import Pose


class ReferenceLine:
    """参考线"""

    __NUM_REFERENCE_LINES: int = 0
    """全局参考线数量"""

    def __init__(self, ctrl_pts: List[Tuple[float, float]]):
        """
        输入控制点坐标初始化参考线
        """
        self.id = ReferenceLine.__NUM_REFERENCE_LINES
        """参考线ID"""
        ReferenceLine.__NUM_REFERENCE_LINES += 1
        self.x_min = math.inf
        """参考线x坐标最小值"""
        self.x_max = -math.inf
        """参考线x坐标最大值"""
        self.y_min = math.inf
        """参考线y坐标最小值"""
        self.y_max = -math.inf
        """参考线y坐标最大值"""
        self.ctrl_pts: np.ndarray = np.array(ctrl_pts)
        """用于生成参考线的控制点坐标"""
        self.pts: List[Pose] = []
        """参考线上的点"""
        self.pt_finder = PointFinder()
        """用于快速查找参考线上的点"""
        self.__genFromCtrlPts()
        self.dist_between_pts = np.full(len(self.pts) - 1, -1.0)
        """参考线上一个点到下一个点的距离，为减少初始化和后续的计算时间，这里采用缓存的方式"""
        self.__close_to_dest_threshold = 1.888 * Config.WHEEL_BASE
        """接近参考线终点的距离阈值，到终点的距离小于该值则认为接近终点"""

    def __genFromCtrlPts(self):
        """基于控制点生成参考线"""
        BSpline.SetK(Config.B_SPLINE_ORDER)
        BSpline.SetIntervalDist(Config.B_SPLINE_INTERVAL_DIST)
        sample_pts = BSpline.GenFromCtrlPts(self.ctrl_pts)
        # 计算每个采样点的斜率并据此计算角度
        angles = np.zeros(sample_pts.shape[0])
        delta_x_array = np.diff(sample_pts[:, 0], append=np.nan)
        delta_y_array = np.diff(sample_pts[:, 1], append=np.nan)
        angles[0] = np.arctan2(delta_y_array[0], delta_x_array[0])
        angles[1] = np.arctan2(delta_y_array[-2], delta_x_array[-2])
        for i in range(1, sample_pts.shape[0] - 1):
            angle0 = np.arctan2(delta_y_array[i - 1], delta_x_array[i - 1])
            angle1 = np.arctan2(delta_y_array[i], delta_x_array[i])
            angles[i] = 0.5 * (angle0 + angle1)
        ref_line: List[Pose] = []
        for i in range(sample_pts.shape[0]):
            x = sample_pts[i][0]
            y = sample_pts[i][1]
            theta = angles[i]
            pose = Pose(x, y, theta)
            ref_line.append(pose)
        # 计算曲率
        curvatures = Curvature.CalAlongCurve(
            [pose.x for pose in ref_line], [pose.y for pose in ref_line]
        )
        for i, pose in enumerate(ref_line):
            pose.setCurvature(curvatures[i])
        # 过滤掉参考线中过于接近的点
        ref_line_filtered: List[Pose] = []
        ref_line_filtered.append(ref_line[0])
        for i in range(1, len(ref_line)):
            pt = ref_line[i]
            if ref_line_filtered[-1].isTooClose(pt):
                continue
            self.x_min = min(self.x_min, pt.x)
            self.x_max = max(self.x_max, pt.x)
            self.y_min = min(self.y_min, pt.y)
            self.y_max = max(self.y_max, pt.y)
            self.pt_finder.addPt(pt.x, pt.y)
            ref_line_filtered.append(pt)
        self.pts = ref_line_filtered
        for i in range(len(self.pts)):
            self.pts[i].ref_line_id = self.id
            self.pts[i].id = i
        self.pt_finder.buildKDTree()
        Log.info("%i reference line generated." % (self.id + 1))

    @property
    def start(self) -> Pose:
        """返回参考线的起点"""
        return self.pts[0]

    @property
    def destination(self) -> Pose:
        """返回参考线的终点"""
        return self.pts[-1]

    def isDestination(self, pose: Pose) -> bool:
        """判断pose是否是参考线的终点"""
        return pose == self.destination

    def isCloseToDestination(self, pose: Pose) -> bool:
        """判断pose是否接近参考线的终点"""
        return pose.calDistance(self.destination) < self.__close_to_dest_threshold

    def getRefPt(self, index: int) -> Pose:
        """根据参考点的序号返回参考点"""
        if index < 0 or index >= len(self.pts):
            Log.warning("Invalid reference point index %d!" % index)
            return self.pts[0]
        return self.pts[index]

    def getDistToNextPt(self, pt_id: int) -> float:
        """返回从start_index开始到下一个参考点的距离"""
        if pt_id < 0 or pt_id >= len(self.pts):
            Log.warning("Invalid point index %d!" % pt_id)
            return 0.0
        elif pt_id == len(self.pts) - 1:
            return 0.0
        else:
            if self.dist_between_pts[pt_id] < -Config.FLOAT_EPS:
                self.dist_between_pts[pt_id] = self.pts[pt_id].calDistance(
                    self.pts[pt_id + 1]
                )
            return self.dist_between_pts[pt_id]

    def getNearestPt(self, x: float, y: float) -> Pose:
        """获取距离(x,y)最近的参考点"""
        index = self.pt_finder.findIndex(x, y)
        return self.getRefPt(index)

    def getNearestPtInRng(
        self, x: float, y: float, start_index: int, stop_index: int = -1
    ) -> Pose:
        """获取在start_index和stop_index之间距离(x,y)最近的参考点"""
        if stop_index < 0:
            stop_index = len(self.pts)
        index = self.pt_finder.findIndexInRng(x, y, start_index, stop_index)
        return self.getRefPt(index)

    def getNearestPtInDist(
        self, x: float, y: float, start_index: int, dist: float
    ) -> Pose:
        """获取从start_index开始，距离[x,y]距离不超过dist的最近参考点"""
        index = self.pt_finder.findIndexInDistance(x, y, start_index, dist)
        return self.getRefPt(index)

    def getPtAfterDistance(self, start_index: int, dist: float) -> Pose:
        """返回从起始点后dist的距离之后的那个点的序号，如果超出范围则返回最后一个点"""
        index = len(self.pts) - 1
        if start_index < 0 or start_index >= len(self.pts):
            Log.warning("Invalid start point index %d!" % start_index)
        elif dist < Config.ZERO_EPS:
            return self.getRefPt(start_index)
        else:
            total_dist = 0.0
            for i in range(start_index, len(self.pts) - 1):
                total_dist += self.getDistToNextPt(i)
                if total_dist >= dist:
                    index = i + 1
                    break
        return self.getRefPt(index)

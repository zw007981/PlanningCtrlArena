# -*- coding: utf-8 -*-


import os
import sys

sys.path.append(os.getcwd())

import math
from typing import List

import numpy as np
from scipy import spatial

from src.util.config import Config


class PointFinder:
    """基于KD tree构建，用于快速找到离一个position最近的地图点的编号"""

    def __init__(self, eps=Config.FLOAT_EPS):
        """基于x，y方向上允许的误差进行初始化"""
        self.eps = eps
        """允许的误差"""
        self.x_list: List[float] = list()
        """存储的地图点的x坐标"""
        self.y_list: List[float] = list()
        """存储的地图点的y坐标"""
        self.positions: np.ndarray
        """存储的地图点的信息，是一个二维数组，每一行是一个地图点的信息，包括x和y坐标"""
        self.has_built_KD_tree = False
        """是否已经成功构建KD tree，已经成功构建是PointFinder能正常工作的必要条件"""
        self.tree: spatial.cKDTree
        """建立的KD tree"""

    def addPt(self, x: float, y: float):
        """添加一个地图点的数据"""
        self.x_list.append(x)
        self.y_list.append(y)
        if self.has_built_KD_tree:
            self.has_built_KD_tree = False

    def buildKDTree(self):
        """构建KD tree"""
        try:
            self.positions = np.array((self.x_list, self.y_list), dtype=float)
            self.positions = self.positions.T
            self.tree = spatial.cKDTree(self.positions)
            self.has_built_KD_tree = True
        except:
            self.has_built_KD_tree = False
            print("Error: Failed to build KDTree!!")
        assert self.has_built_KD_tree == True

    def findIndex(self, x: float, y: float) -> int:
        """找到一个离position最近的地图点对应的index"""
        # 如果还没有构建KDTree则返回-1。
        if not self.has_built_KD_tree:
            print("Warning: KDTree has not been built!")
            return -1
        else:
            # 找到离[x,y]最近的点的编号。
            _, index = self.tree.query((x, y))  # type: ignore
            return index
            # nearest_point = self.positions[index, :]
            # # 我们要求最近的点离[x,y]的距离要在误差允许的范围内。
            # if (
            #     abs(nearest_point[0] - x) < self.eps
            #     and abs(nearest_point[1] - y) < self.eps
            # ):
            #     return index
            # else:
            #     return -1

    def findIndexInRng(
        self, x: float, y: float, start_index: int, stop_index: int
    ) -> int:
        """找到从start_index到stop_index之间离position最近地图点的编号"""
        if not self.has_built_KD_tree:
            print("Warning: KDTree has not been built!")
            return -1
        else:
            _, indices = self.tree.query((x, y), k=len(self.positions))
            # 过滤掉不在范围内的点
            for idx in indices:
                if start_index <= idx < stop_index:
                    return idx
            return -1

    def findIndexInDistance(
        self, x: float, y: float, start_index: int, dist: float
    ) -> int:
        """寻找从start_index开始，距离[x,y]距离不超过dist的最近地图点的编号"""
        if not self.has_built_KD_tree:
            print("Warning: KDTree has not been built!")
            return -1
        else:
            _, indices = self.tree.query((x, y), k=len(self.positions))
            for index in indices:
                if start_index <= index:
                    dist_to_pt = math.hypot(
                        self.x_list[index] - x, self.y_list[index] - y
                    )
                    if dist_to_pt <= dist:
                        return index
                    return start_index
            return start_index

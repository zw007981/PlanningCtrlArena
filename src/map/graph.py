# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import Any, Dict, List, Optional

from src.map.point_finder import PointFinder
from src.util.config import Config
from src.util.log import Log
from src.util.position import Position


class Node:
    """图中的节点"""

    def __init__(self, pos: Position, id: int = -1):
        self.pos = pos
        """节点的位置"""
        self.id = id
        """节点ID"""
        self.children: List["Node"] = []
        """子节点列表"""

    def setChildren(self, children: List["Node"]):
        """设置子节点"""
        self.children = children


class Graph:
    """图数据结构"""

    def __init__(self, nodes_info: List[Dict[str, Any]]):
        """读取配置文件根据其中的节点信息初始化图"""
        self.nodes: List[Node] = []
        """节点列表"""
        self.pt_finder = PointFinder()
        """用于快速查找节点"""
        try:
            self.__genNodes(nodes_info)
            self.__genEdges(nodes_info)
        except Exception as e:
            Log.error("Failed to generate graph: {}!!!".format(e))
            raise e

    @property
    def size(self) -> int:
        """获取图中节点的数量"""
        return len(self.nodes)

    def getNodeFromID(self, id: int) -> Node:
        """根据节点ID获取节点"""
        if id < 0 or id >= len(self.nodes):
            Log.error("Node ID {} out of range!!!".format(id))
            raise ValueError("Node ID out of range!!!")
        return self.nodes[id]

    def getNodeFromPos(self, x: float, y: float) -> Optional[Node]:
        """根据坐标获取节点，如果离最近的节点距离小于阈值则返回该节点，否则返回None"""
        nearest_node_index = self.pt_finder.findIndex(x, y)
        if (
            nearest_node_index >= 0
            and Position(x, y) == self.nodes[nearest_node_index].pos
        ):
            return self.nodes[nearest_node_index]
        return None

    def __isNodeDuplicate(self, x: float, y: float) -> bool:
        """判断一个坐标是否已经存在于节点列表中"""
        return self.getNodeFromPos(x, y) is not None

    def __genNodes(self, nodes_info: List[Dict[str, Any]]):
        """根据节点信息生成节点列表"""
        for node_info in nodes_info:
            x, y = node_info["position"]["x"], node_info["position"]["y"]
            if self.__isNodeDuplicate(x, y):
                Log.error(
                    "Node at position ({:.3f}, {:.3f}) already exists!!!".format(x, y)
                )
                raise ValueError("Node already exists!!!")
            node = Node(Position(x, y), id=self.size)
            self.nodes.append(node)
            self.pt_finder.addPt(x, y)
            self.pt_finder.buildKDTree()

    def __genEdges(self, nodes_info: List[Dict[str, Any]]):
        """根据节点信息生成节点之间的边"""
        for parent_node_index, parent_node_info in enumerate(nodes_info):
            parent_node = self.getNodeFromID(parent_node_index)
            children: List[Node] = []
            if "children" in parent_node_info:
                for child_info in parent_node_info["children"]:
                    child_x, child_y = child_info["x"], child_info["y"]
                    child_node = self.getNodeFromPos(child_x, child_y)
                    if child_node is None:
                        Log.error(
                            "Child node at position ({:.3f}, {:.3f}) does not exist!!!".format(
                                child_x, child_y
                            )
                        )
                        raise ValueError("Child node does not exist!!!")
                    else:
                        children.append(child_node)
            parent_node.setChildren(children)

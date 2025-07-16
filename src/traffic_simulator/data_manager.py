# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import copy
import json
from typing import Any, Dict, List

from src.equipment.equipment_manager import EquipmentManager
from src.kinematic_model.ctrl_input import CtrlInput
from src.map.map import Map
from src.map.reference_line import ReferenceLine
from src.util.config import Config
from src.util.log import Log
from src.util.pose import Pose


class DataManager:
    """数据管理器，用于管理仿真器中的数据，例如地图，车辆信息"""

    def __init__(self, config_name: str = "config.json") -> None:
        """使用配置文件初始化数据管理器"""
        self.map: Map = Map()
        """地图"""
        self.map.loadMapFromFile(config_name)
        self.equip_mgr = EquipmentManager()
        """设备管理器"""
        self.equip_mgr.loadEquipmentFromFile(config_name)
        # 一般来说配置中至少存在一辆车辆的初始信息，把第一辆车辆视为主车
        if self.equip_mgr.size == 0:
            Log.error("No car in the configuration file!!!")
            raise ValueError("No car in the configuration file!!!")
        self.ego_car_id = sorted(self.equip_mgr.id_to_status.keys())[0]
        """主车ID"""
        self.ego_car = self.equip_mgr.get(self.ego_car_id)
        """主车信息"""
        self.id_to_ref_line: Dict[str, ReferenceLine] = {}
        """车辆ID到参考线的映射，初始化后就不会改变"""
        self.id_to_destination: Dict[str, Pose] = {}
        """车辆ID到目的地的映射，初始化后就不会改变"""
        for car_id, car in self.equip_mgr.id_to_status.items():
            ref_line = self.map.getRefLine(car.tgt_ref_line_id)
            self.id_to_ref_line[car_id] = ref_line
            self.id_to_destination[car_id] = copy.deepcopy(ref_line.destination)
        self.id_to_ref_pt: Dict[str, Pose] = {}
        """车辆ID到参考点的映射"""
        self.id_to_ctrl_input: Dict[str, CtrlInput] = {}
        """车辆ID到控制输入的映射"""
        for car_id in self.car_id_list:
            self.id_to_ctrl_input[car_id] = CtrlInput(
                kinematic_model_type="BICYCLE_MODEL"
            )
        self.__initRefPts()

    @property
    def car_id_list(self) -> List[str]:
        """车辆ID列表"""
        return sorted(self.equip_mgr.id_to_status.keys())

    def __initRefPts(self) -> None:
        """初始化每辆车的参考点"""
        for car_id, ref_line in self.id_to_ref_line.items():
            car = self.equip_mgr.get(car_id)
            ref_pt = self.map.getNearestRefPt(car.pose.x, car.pose.y, ref_line.id)
            self.id_to_ref_pt[car_id] = ref_pt

    def __updateRefPts(self) -> None:
        """更新每辆车的参考点，要求更新后参考点的序号不能小于更新前"""
        for car_id, ref_line in self.id_to_ref_line.items():
            car = self.equip_mgr.get(car_id)
            ref_pt = self.map.getNearestRefPtInRng(
                car.pose.x, car.pose.y, ref_line.id, self.id_to_ref_pt[car_id].id
            )
            self.id_to_ref_pt[car_id] = ref_pt

    def __resetCars(self) -> None:
        """重置车辆状态"""
        for car_id in self.car_id_list:
            self.equip_mgr.get(car_id).resetState()

    def __applyControlInputsToCars(self) -> None:
        """使用控制输入更新车辆状态"""
        for car_id, ctrl_input in self.id_to_ctrl_input.items():
            self.equip_mgr.get(car_id).updateState(ctrl_input)

    def update(self) -> None:
        """根据控制输入更新车辆状态和参考点信息"""
        self.__applyControlInputsToCars()
        self.__updateRefPts()

    def reset(self) -> None:
        """重置车辆和参考信息，一般在重新开始仿真时调用"""
        self.__resetCars()
        self.__initRefPts()
        for obstacle in self.map.obstacles:
            obstacle.has_been_observed = False

    def saveMotionDataToFile(self, file_name: str = "motion_data.json") -> None:
        """将设备运动数据保存到文件"""
        if self.equip_mgr.size == 0:
            Log.warning("No cars to save motion data!")
            return
        file_path = os.path.join(Config.LOG_DIR, file_name)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        data_info: Dict[str, Any] = {}
        data_info["cars"] = []
        for _, car in self.equip_mgr.id_to_status.items():
            data_info["cars"].append(car.genMotionData())
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data_info, file, ensure_ascii=False, indent=4)
        Log.info("Motion data saved to {}.".format(file_path))

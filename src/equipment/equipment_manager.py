# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import json
from typing import Any, Dict

from src.algo.level_k.level import Level
from src.equipment.car import Car
from src.util.config import Config
from src.util.log import Log
from src.util.pose import Pose


class EquipmentManager:
    """设备管理器，用于更新和管理设备的状态"""

    def __init__(self):
        self.id_to_status: Dict[str, Car] = {}
        """从车辆ID到车辆状态的映射"""

    def __clear(self) -> None:
        """清空设备状态"""
        self.id_to_status.clear()

    @property
    def size(self) -> int:
        """获取设备数量"""
        return len(self.id_to_status)

    def get(self, id: str) -> Car:
        """获取设备状态"""
        return self.id_to_status[id]

    def loadEquipmentFromFile(self, file_name: str) -> None:
        """从文件中加载设备信息"""
        self.__clear()
        # 读取控制点信息
        file_path = os.path.join(Config.GetConfigFolder(), file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if "cars" not in data:
                    raise ValueError("No cars in the file")
                for car in data["cars"]:
                    car_id = car["id"]
                    start_pose_ = car["start_pose"]
                    start_pose = Pose(
                        start_pose_["x"], start_pose_["y"], start_pose_["theta"]
                    )
                    self.id_to_status[car_id] = Car(
                        car_id,
                        start_pose,
                        car["v"],
                        car.get("kinematic_model_type", Config.KINEMATIC_MODEL_TYPE),
                        int(car.get("target_reference_line_id", 0)),
                        (
                            Level.GetLevel(car["level"])
                            if "level" in car
                            else Level.LEVEL_0
                        ),
                    )

    def saveMotionDataToFile(self, file_name: str = "motion_data.json") -> None:
        """将设备运动数据保存到文件"""
        if len(self.id_to_status) == 0:
            Log.warning("No cars to save motion data.")
            return
        file_path = os.path.join(Config.LOG_DIR, file_name)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        data_info: Dict[str, Any] = {}
        data_info["cars"] = []
        for _, car in self.id_to_status.items():
            data_info["cars"].append(car.genMotionData())
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data_info, file, ensure_ascii=False, indent=4)
        Log.info("Motion data saved to {}.".format(file_path))

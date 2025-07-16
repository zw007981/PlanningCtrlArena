# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import json
from typing import List

from matplotlib import pyplot as plt

from src.util.log import Log


class Data:
    """数据类，用于存储数据"""

    def __init__(
        self,
        Pn: float,
        time: List[float],
        x: List[float],
        y: List[float],
        theta: List[float],
        v: List[float],
        a: List[float],
        delta: List[float],
    ):
        self.Pn = Pn
        """Pn值"""
        self.time = time
        """时间"""
        self.x = x
        """x坐标"""
        self.y = y
        """y坐标"""
        self.theta = theta
        """角度"""
        self.v = v
        """速度"""
        self.a = a
        """加速度"""
        self.delta = delta
        """前轮转角"""


class CMPCInsight:
    """分析CMPC算法的工具类，主要是不同Pn值对算法性能的影响"""

    def __init__(self):
        self.folder_path = os.path.join("data", "CMPC")
        """需要分析的文件夹路径"""
        if not os.path.exists(self.folder_path):
            Log.error("Data folder {} does not exist!!!".format(self.folder_path))
            raise ValueError("Data folder does not exist!!!")
        file_names = self.__getFilesToAnalyze()
        self.dataset: List[Data] = [
            self.__extractData(file_name) for file_name in file_names
        ]
        """需要分析的数据集"""
        self.dataset.sort(key=lambda x: x.Pn)

    def __getFilesToAnalyze(self) -> List[str]:
        """获取需要分析的文件"""
        file_names = [
            file_name
            for file_name in os.listdir(self.folder_path)
            if file_name.endswith(".json")
        ]
        if len(file_names) == 0:
            Log.error(
                "There is no file in folder {} to analyze!!!".format(self.folder_path)
            )
            raise ValueError("No file to analyze!!!")
        else:
            Log.info(
                "There are {} files in folder {} to analyze.".format(
                    len(file_names), self.folder_path
                )
            )
            return file_names

    def __extractData(self, file_name: str) -> Data:
        """
        提取文件中的数据并返回，
        数据内容包含：Pn值、time, x, y, theta, v, a, delta
        """
        try:
            Log.info("Extracting data from file {}.".format(file_name))
            # 从文件名中提取Pn值，文件名类似于Pn_0.2.json
            Pn = float(file_name.split("_")[1].split(".j")[0])
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, "r") as f:
                json_data = json.load(f)
                car_info = json_data["cars"][0]
                return Data(
                    Pn,
                    car_info["time"],
                    car_info["x"],
                    car_info["y"],
                    car_info["theta"],
                    car_info["v"],
                    car_info["a"],
                    car_info["delta"],
                )
        except Exception as e:
            Log.error(
                "Error occurred while extracting data from file {}: {}".format(
                    file_name, e
                )
            )
            raise e

    def genReport(self):
        """生成报告"""
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("CMPC Insight")
        # 第一幅子图绘制轨迹图
        for data in self.dataset:
            axs[0].plot(data.x, data.y, label="Pn={:.3f}".format(data.Pn))
        axs[0].set_title("trajectory")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].legend()
        axs[0].grid(True)
        # 第二幅子图绘制x坐标与前轮转角的关系
        for data in self.dataset:
            # 在图例中标注出极值出现的位置
            label_str = "Pn={:.3f}".format(data.Pn)
            max_delta_index = data.delta.index(max(data.delta))
            label_str += ", max_delta={:.3f} at x={:.3f}".format(
                data.delta[max_delta_index], data.x[max_delta_index]
            )
            axs[1].plot(data.x[:-1], data.delta, label=label_str.format(data.Pn))
        axs[1].set_title("x vs delta")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("delta")
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.folder_path, "CMPC_report.png"))
        Log.info(
            "Report saved to {}.".format(
                os.path.join(self.folder_path, "CMPC_report.png")
            )
        )
        plt.show()


if __name__ == "__main__":
    insight = CMPCInsight()
    insight.genReport()

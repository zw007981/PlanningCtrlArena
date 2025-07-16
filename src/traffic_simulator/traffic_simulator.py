# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from abc import ABC, ABCMeta, abstractmethod

from matplotlib import pyplot as plt
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit

from src.equipment.car import Car
from src.kinematic_model.ctrl_input import CtrlInput
from src.map.reference_line import ReferenceLine
from src.traffic_simulator.color import *
from src.traffic_simulator.data_manager import DataManager
from src.traffic_simulator.graphic_item_manager import GraphicItemManager
from src.traffic_simulator.simulator_tool import SimulatorTool
from src.traffic_simulator.ui.ui_MainGui import Ui_MainWindow
from src.util.config import Config
from src.util.log import Log
from src.util.pose import Pose
from src.util.timer import Timer as PerfTimer


class MetaQMainWindowABC(type(QMainWindow), ABCMeta):  # type: ignore
    """创建一个自定义的元类，继承所有基类的元类"""

    pass


class TrafficSimulator(QMainWindow, Ui_MainWindow, ABC, metaclass=MetaQMainWindowABC):
    """交通模拟器，用于可视化交通状况以支持规划控制算法的调试"""

    import pyqtgraph as pg

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    def __init__(self, config_name="config.json"):
        """利用配置文件初始化交通仿真器"""
        super().__init__()
        self.setupUi(self)
        Log.setLogWidget(self.information_browser2)
        self.graph_widget.setAspectLocked(lock=True, ratio=1)  # type: ignore
        self.graph_widget.addLegend()
        self.information_browser1.setReadOnly(True)
        self.information_browser1.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.data_mgr = DataManager(config_name)
        """数据管理器"""
        self.graphic_item_mgr = GraphicItemManager(self.data_mgr, self.graph_widget)
        """图形项管理器"""
        self.sim_tool = SimulatorTool(type(self).__name__.removesuffix("Demo"))
        """仿真工具"""
        self.timer = QtCore.QTimer()
        """模拟计时器"""
        self.sim_time = 0.0
        """当前模拟时间"""
        self.ratio: float = 1.0
        """模拟速率"""
        self.is_running = False
        """是否正在运行"""
        self.is_finished = False
        """模拟是否完成"""
        self.__last_info_browser1_text = ""
        """上次information_browser1的文本内容"""
        self.__initUI()

    @property
    def ego_car(self) -> Car:
        """主车"""
        return self.data_mgr.ego_car

    @property
    def perf_timer(self) -> PerfTimer:
        """性能计时器"""
        return self.sim_tool.perf_timer

    @abstractmethod
    def update(self) -> None:
        """更新方法，通常要求基于当前帧信息生成控制输入，需要在子类重写"""
        Log.error("Please override the update method in the subclass!!!")
        pass

    def checkIfCarExists(self, car_id: str) -> None:
        """检查车辆是否存在，注意如果不存在会直接抛出异常"""
        if car_id not in self.data_mgr.equip_mgr.id_to_status:
            Log.error("Car ID {} not found!!!".format(car_id))
            raise ValueError("Car ID not found!!!")

    def getCar(self, car_id: str = "") -> Car:
        """获取车辆信息，默认获取主车"""
        car_id = car_id or self.data_mgr.ego_car_id
        return self.data_mgr.equip_mgr.get(car_id)

    def getRefLine(self, car_id: str = "") -> ReferenceLine:
        """
        获取车辆的参考线，默认获取主车的，
        包括这里在内的获取信息的方法全都不对车辆id做校验，
        因为Python中查询字典中是否存在键较为耗时。
        """
        car_id = car_id or self.data_mgr.ego_car_id
        return self.data_mgr.id_to_ref_line[car_id]

    def getDestination(self, car_id: str = "") -> Pose:
        """获取车辆的目的地，默认获取主车的"""
        car_id = car_id or self.data_mgr.ego_car_id
        return self.data_mgr.id_to_destination[car_id]

    def getDistToDest(self, car_id: str = "") -> float:
        """获取车辆距离目的地的距离，默认获取主车的"""
        car_id = car_id or self.data_mgr.ego_car_id
        destination = self.data_mgr.id_to_destination[car_id]
        car = self.data_mgr.equip_mgr.get(car_id)
        return destination.calDistance(car.pose)

    def getRefPt(self, car_id: str = "") -> Pose:
        """获取车辆的参考点，默认获取主车的"""
        car_id = car_id or self.data_mgr.ego_car_id
        return self.data_mgr.id_to_ref_pt[car_id]

    def getCtrlInput(self, car_id: str = "") -> CtrlInput:
        """获取车辆的控制输入，默认获取主车的"""
        car_id = car_id or self.data_mgr.ego_car_id
        return self.data_mgr.id_to_ctrl_input[car_id]

    def finalize(self) -> None:
        """仿真完成后的处理方法，可以根据需要在子类重写"""
        self.is_running = False
        self.is_finished = True
        self.perf_timer.printAveTime()

    def restart(self) -> None:
        """重新开始模拟仿真的方法，可以根据需要在子类重写"""
        pass

    def genReport(self) -> None:
        """生成报告的方法，默认生成主车的标准报告，可以根据需要在子类重写"""
        car = self.ego_car
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        data = [
            (
                car.time_sequence[:-1],
                [ctrl[0] for ctrl in car.ctrl_input_history],
                "r",
                "Acceleration",
            ),
            (
                car.time_sequence[:-1],
                [ctrl[1] for ctrl in car.ctrl_input_history],
                "b",
                "Steering Angle",
            ),
            (car.time_sequence, car.vel_history, "c", "Speed"),
            (
                car.time_sequence[:-1],
                [1000 * t for t in self.perf_timer.time_list],
                "m",
                "Computation Time(ms)",
            ),
        ]
        for ax, (x, y, color, ylabel) in zip(axs.flat, data):
            ax.plot(x, y, color)
            ax.set(title=f"Time vs {ylabel}", xlabel="Time", ylabel=ylabel)
            ax.grid(True)
        plt.tight_layout()
        report_file_path = os.path.join(Config.LOG_DIR, "report.png")
        fig.savefig(report_file_path)
        Log.info(f"Report saved to {report_file_path}.")
        plt.show()

    def closeEvent(self, event: QtCore.QEvent) -> None:
        """处理关闭事件"""
        self.data_mgr.saveMotionDataToFile()
        self.sim_tool.genGif()
        event.accept()

    def __updateView(self) -> None:
        """随着模拟的进行更新视图"""
        if self.is_running and not self.is_finished:
            self.sim_time += Config.DELTA_T
            self.data_mgr.update()
            self.graphic_item_mgr.update(self.data_mgr)
            with self.perf_timer:
                self.update()
            self.sim_tool.update(self.graph_widget)
            self.updateInfoBrowser1(
                "pose={}, v={:.2f}\n{}".format(
                    self.ego_car.pose, self.ego_car.v, self.getCtrlInput()
                )
            )
            self.__updateInfoBrowser3()

    def __restartView(self) -> None:
        """重新开始模拟后更新状态"""
        self.is_running = False
        self.is_finished = False
        self.sim_time = 0.0
        self.data_mgr.reset()
        self.graphic_item_mgr.reset(self.data_mgr)
        self.restart()
        self.sim_tool.reset()
        Log.info("Simulation restarted successfully.")
        self.updateInfoBrowser1("")
        self.__updateInfoBrowser3()

    def __initUI(self) -> None:
        """初始化UI"""
        self.stop_or_start_button.clicked.connect(self.__runOrPause)
        self.timer.timeout.connect(self.__updateView)
        self.timer.start(int(Config.DELTA_T * 1000))
        self.information_browser1.setReadOnly(True)
        self.information_browser1.setStyleSheet("color: green")
        self.information_browser3.setReadOnly(True)
        self.information_browser3.setStyleSheet("color: blue")
        self.__updateInfoBrowser3()
        self.decelerate_button.clicked.connect(self.__decelerate)
        self.accelerate_button.clicked.connect(self.__accelerate)
        self.restart_button.clicked.connect(self.__restartView)
        self.generate_report_button.clicked.connect(self.__genReport)

    def __decelerate(self) -> None:
        """减速"""
        if self.ratio > 1 / 16:
            self.ratio /= 2
            Config.DELTA_T = Config.INIT_DELTA_T * self.ratio
            Log.debug("Simulation speed decreased.")
            self.__updateInfoBrowser3()

    def __accelerate(self) -> None:
        """加速"""
        if self.ratio < 16:
            self.ratio *= 2
            Config.DELTA_T = Config.INIT_DELTA_T * self.ratio
            Log.debug("Simulation speed increased.")
            self.__updateInfoBrowser3()

    def __runOrPause(self) -> None:
        """开始运行或者暂停"""
        if self.is_finished:
            Log.warning("Cannot start a finished simulation!")
        elif self.is_running:
            self.is_running = False
            Log.info("Simulation paused.")
        else:
            self.is_running = True
            Log.info("Simulation started.")

    def __genReport(self) -> None:
        """模拟完成后生成报告"""
        if self.is_running:
            Log.warning("Cannot generate report during simulation!")
        elif self.sim_time < Config.ZERO_EPS:
            Log.warning("Cannot generate report, the simulation has not started!")
        else:
            Log.info("Generating simulation report...")
            self.genReport()

    def updateInfoBrowser1(self, msg: str = "") -> None:
        """更新1号信息窗口"""
        if self.__last_info_browser1_text != msg:
            self.__last_info_browser1_text = msg
            self.information_browser1.setPlainText(msg)

    def __updateInfoBrowser3(self) -> None:
        """更新状态窗口"""
        msg_to_show: str = "Speed: X{:.2f}\nTime: {:.2f}s".format(
            self.ratio, self.sim_time
        )
        if self.information_browser3.toPlainText() != msg_to_show:
            self.information_browser3.setText(msg_to_show)


def runTrafficSimulator():
    """运行交通模拟器"""
    app = QApplication(sys.argv)
    traffic_simulator = TrafficSimulator()  # type: ignore
    traffic_simulator.show()
    app.exec()


if __name__ == "__main__":
    runTrafficSimulator()

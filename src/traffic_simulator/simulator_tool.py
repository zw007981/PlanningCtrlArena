# -*- coding: utf-8 -*-

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import imageio.v3 as iio
import numpy as np
from pyqtgraph import PlotWidget

from src.util.config import Config
from src.util.log import Log
from src.util.timer import Timer


class SimulatorTool:
    """仿真工具类"""

    def __init__(self, program_name: str = "") -> None:
        """初始化仿真工具，可以选择性地输入程序名以更好地初始化性能计时器"""
        self.frame_buffer = []
        """帧缓存，用于存储每一帧的数据以生成 GIF"""
        self.executor = ThreadPoolExecutor(max_workers=1)
        """线程池，把帧数据的处理放到后台线程中避免过多地干扰主线程"""
        self.perf_timer = Timer(program_name)
        Log.debug("SimulatorTool initialized.")

    def update(self, graph_widget: PlotWidget) -> None:
        """随着仿真器运行捕获当前帧并存储"""
        frame = graph_widget.grab()
        # 将转换和存储过程提交到线程池执行
        self.executor.submit(self.processFrame, frame)

    def processFrame(self, frame) -> None:
        """在后台线程中转换并存储帧数据"""
        image = frame.toImage()
        ptr = image.bits().tobytes()  # type: ignore
        image_array = np.frombuffer(ptr, dtype=np.uint8).reshape(
            image.height(), image.width(), 4
        )
        self.frame_buffer.append(image_array)

    def reset(self) -> None:
        """重置仿真工具：清空帧缓存并重置性能计时器。"""
        self.frame_buffer.clear()
        self.perf_timer.clear()
        Log.debug("Frame buffer and performance timer cleared.")

    def setPerfTimerName(self, name: str) -> None:
        """设置性能计时器名称"""
        if name and name != self.perf_timer.name:
            self.perf_timer = Timer(name)

    def genGif(self) -> None:
        """
        基于帧缓存生成 GIF：
        1. 等待所有后台线程任务结束；
        2. 调用 imageio 写入 GIF 文件。
        """
        if not self.frame_buffer:
            Log.warning("No frame captured!")
            return

        Log.info("Generating GIF...")
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        gif_path = os.path.join(Config.LOG_DIR, "simulation_animation.gif")
        # 关闭线程池，等待所有任务完成
        self.executor.shutdown(wait=True)
        # 利用 imageio 生成 GIF
        iio.imwrite(
            gif_path, self.frame_buffer, duration=1000 * (Config.DELTA_T), loop=0
        )
        Log.info("Gif saved to {}.".format(gif_path))

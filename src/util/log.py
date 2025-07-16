# -*- coding: utf-8 -*-

import datetime
import inspect
import os
import sys
from enum import Enum
from typing import Optional, TextIO

sys.path.append(os.getcwd())


from PySide6 import QtWidgets

from src.traffic_simulator.color import *


class LogLevel(Enum):
    """日志级别"""

    DEBUG = 0
    """调试"""
    INFO = 1
    """信息"""
    WARNING = 2
    """警告"""
    ERROR = 3
    """错误"""


class Log:
    """日志类，此处主要是为了把日志输出到ui中的日志输出窗口"""

    LOG_WIDGET: Optional[QtWidgets.QTextBrowser] = None
    """日志输出窗口"""
    LOG_FILE_PATH: str = "log/log.log"
    """日志文件路径"""
    LOG_FILE: Optional[TextIO] = None
    """日志文件对象"""
    HAS_LOGGED: bool = False
    """是否已经记录过日志"""

    @staticmethod
    def __getTimeStr() -> str:
        """获取当前时间对应的字符串"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    @staticmethod
    def __getCaller() -> str:
        """获取调用日志的文件名和行号"""
        dirname: str = "unknown"
        basename: str = "unknown"
        lineno: int = 0
        frame = inspect.stack()[3]
        module = inspect.getmodule(frame[0])
        if module:
            file_name = module.__file__
            dirname = os.path.basename(os.path.dirname(str(file_name)))
            basename = os.path.basename(str(file_name))
            lineno = frame.lineno
        return "%s/%s-%d" % (dirname, basename, lineno)

    @staticmethod
    def __logMsgToFile(msg: str, lvl: LogLevel = LogLevel.INFO) -> None:
        os.makedirs(os.path.dirname(Log.LOG_FILE_PATH), exist_ok=True)
        if not Log.HAS_LOGGED and os.path.exists(Log.LOG_FILE_PATH):
            os.remove(Log.LOG_FILE_PATH)
        Log.HAS_LOGGED = True
        if Log.LOG_FILE == None:
            Log.LOG_FILE = open(Log.LOG_FILE_PATH, "a", encoding="utf-8")
            Log.info("Log file created: %s." % Log.LOG_FILE_PATH)
        msg_to_log = "%s [%s] %s: %s" % (
            Log.__getTimeStr(),
            lvl.name,
            Log.__getCaller(),
            msg,
        )
        Log.LOG_FILE.write(msg_to_log + "\n")
        Log.LOG_FILE.flush()
        if Log.LOG_WIDGET is None:
            print(msg_to_log)

    @staticmethod
    def __logMsgToLogWidget(
        log_widget: QtWidgets.QTextBrowser, msg: str, lvl: LogLevel = LogLevel.INFO
    ) -> None:
        """将日志信息打印到日志输出窗口"""
        if lvl == LogLevel.DEBUG:
            log_widget.setTextColor(GREY)
        elif lvl == LogLevel.INFO:
            log_widget.setTextColor(BLACK)
        elif lvl == LogLevel.WARNING:
            log_widget.setTextColor(YELLOW)
        elif lvl == LogLevel.ERROR:
            log_widget.setTextColor(RED)
        msg_to_log = "%s: %s" % (Log.__getTimeStr(), msg)
        log_widget.append(msg_to_log)
        log_widget.verticalScrollBar().setValue(
            log_widget.verticalScrollBar().maximum()
        )
        QtWidgets.QApplication.processEvents()

    @staticmethod
    def setLogWidget(log_widget: QtWidgets.QTextBrowser) -> None:
        """设置日志输出窗口"""
        Log.LOG_WIDGET = log_widget
        Log.info("Log widget set.")

    @staticmethod
    def debug(msg: str) -> None:
        """打印调试信息"""
        if Log.LOG_WIDGET is not None:
            Log.__logMsgToLogWidget(Log.LOG_WIDGET, msg, LogLevel.DEBUG)
        Log.__logMsgToFile(msg, LogLevel.DEBUG)

    @staticmethod
    def info(msg: str) -> None:
        """打印信息"""
        if Log.LOG_WIDGET is not None:
            Log.__logMsgToLogWidget(Log.LOG_WIDGET, msg, LogLevel.INFO)
        Log.__logMsgToFile(msg, LogLevel.INFO)

    @staticmethod
    def warning(msg: str) -> None:
        """打印警告信息"""
        if Log.LOG_WIDGET is not None:
            Log.__logMsgToLogWidget(Log.LOG_WIDGET, msg, LogLevel.WARNING)
        Log.__logMsgToFile(msg, LogLevel.WARNING)

    @staticmethod
    def error(msg: str) -> None:
        """打印错误信息"""
        if Log.LOG_WIDGET is not None:
            Log.__logMsgToLogWidget(Log.LOG_WIDGET, msg, LogLevel.ERROR)
        Log.__logMsgToFile(msg, LogLevel.ERROR)

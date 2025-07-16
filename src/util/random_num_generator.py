# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())


from enum import Enum
from typing import Any, List, Type, TypeVar, Union

import numpy as np

T = TypeVar("T", bound=Enum)


class RandomNumGenerator:
    """随机数生成器"""

    def __init__(self, random_seed: int = 666) -> None:
        self.generator = np.random.RandomState(random_seed)

    def getRandomNum(self, low: int = 0, high: int = 666666) -> int:
        """获取一个随机数，范围是[low, high)"""
        return self.generator.randint(low, high)

    def getRandomFloat(self, low: float = 0.0, high: float = 1.0) -> float:
        """获取一个随机浮点数"""
        return self.generator.uniform(low, high)

    def getRandomEnum(self, enum_class: Type[T]) -> T:
        """从一个Enum类中随机获取一个枚举值"""
        enum_list = np.array(list(enum_class))
        return self.generator.choice(enum_list, 1, replace=False)[0]

    def choice(
        self,
        list_to_choose: Union[List[Any], np.ndarray],
        size: int,
        replace: bool = False,
    ) -> np.ndarray:
        """从列表中随机选择size个元素"""
        return self.generator.choice(list_to_choose, size, replace)


if __name__ == "__main__":
    from src.util.log import Log

    generator = RandomNumGenerator()
    for _ in range(2):
        Log.info("Random number: {}.".format(generator.getRandomNum()))
        Log.info("Random float: {}.".format(generator.getRandomFloat()))
    list_to_choose = [1, 2, 3, 4, 5]
    Log.info("Random choice: {}.".format(generator.choice(list_to_choose, 2)))
    array_to_choose = np.array([1, 2, 3, 4, 5])
    Log.info("Random choice: {}.".format(generator.choice(array_to_choose, 4, True)))

    class State(Enum):
        LEFT = 0
        RIGHT = 1

    sample = generator.getRandomEnum(State)
    Log.info("Random enum: {}.".format(sample))

# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.util.config import Config
from src.util.log import Log
from src.util.tiger import Action, AlphaVector, Belief, Observation, State, Tiger


class VectorSet:
    """alpha向量集合类，存储t时刻的所有alpha向量"""

    def __init__(self, t: int):
        """使用时刻t初始化"""
        self.t = t
        """直到结束还有多少步"""
        self.alpha_vectors: List[AlphaVector] = []
        """alpha向量列表"""

    @property
    def size(self) -> int:
        """集合中alpha向量的数量"""
        return len(self.alpha_vectors)

    def clear(self):
        """清空alpha向量列表"""
        self.alpha_vectors.clear()

    def add(self, alpha_vector: AlphaVector):
        """添加alpha向量"""
        self.alpha_vectors.append(alpha_vector)
        if self.size > Action.Size():
            Log.error(
                "The size of alpha vectors {} exceeds the size of action space {}!!!".format(
                    self.size, Action.Size()
                )
            )
            raise ValueError("Alpha vector size exceeds action space size!!!")

    def prune(self) -> None:
        """剪枝，删除被支配的alpha向量"""
        if not self.alpha_vectors:
            Log.warning("Alpha vector set is empty!")
            return
        dominated_indices: List[int] = []
        for i in range(self.size):
            for j in range(self.size):
                if i != j and self.alpha_vectors[i].isDominated(self.alpha_vectors[j]):
                    dominated_indices.append(i)
        self.alpha_vectors = [
            self.alpha_vectors[i]
            for i in range(self.size)
            if i not in dominated_indices
        ]

    def max(self, state: State) -> AlphaVector:
        """计算给定状态下所有alpha向量中值最大的向量"""
        if not self.alpha_vectors:
            Log.error("Alpha vector set is empty!!!")
            raise ValueError("Alpha vector set is empty!!!")
        return max(self.alpha_vectors, key=lambda vec: vec.get(state))

    def getBestVec(self, belief: Belief) -> AlphaVector:
        """计算给定信念状态下Q值最大的向量"""
        max_val = float("-inf")
        opt_vec_index = -1
        for alpha_vec in self.alpha_vectors:
            val = alpha_vec.dot(belief)
            if val > max_val + Config.FLOAT_EPS:
                max_val = val
                opt_vec_index = self.alpha_vectors.index(alpha_vec)
        if opt_vec_index == -1:
            Log.error("No optimal action found!!!")
            raise ValueError("No optimal action found!!!")
        return self.alpha_vectors[opt_vec_index]

    def getBestAction(self, belief: Belief) -> Action:
        """计算给定信念状态下最优动作"""
        best_vec = self.getBestVec(belief)
        return best_vec.action

    def getVal(self, belief: Belief) -> float:
        """计算给定信念状态下的值函数"""
        best_vec = self.getBestVec(belief)
        return best_vec.dot(belief)


class AlphaVectorSolver:
    """alpha-vectors算法求解器"""

    def __init__(self, env: Tiger, horizon: int = 3, gamma: float = 0.95):
        """使用环境，迭代步数和折扣因子初始化"""
        self.env = env
        """所处环境"""
        self.horizon = horizon
        """迭代步数"""
        self.gamma: float = gamma
        """折扣因子"""
        self.t_to_vec_sets: Dict[int, VectorSet] = {
            t: VectorSet(t) for t in range(1, self.horizon + 1)
        }
        """从时间步到对应的alpha向量集合的映射"""

    def solve(self):
        """基于alpha-vectors算法求解POMDP问题"""
        Log.info("Start solving POMDP problem using alpha-vectors algorithm...")
        self.t_to_vec_sets[1] = self.__genInitVecSet()
        for t in range(2, self.horizon + 1):
            self.t_to_vec_sets[t] = self.__backup(t)
        Log.info("Solving POMDP problem using alpha-vectors algorithm finished.")

    def __genInitVecSet(self) -> VectorSet:
        """生成初始向量集合"""
        vec_set = VectorSet(0)
        for action in Action:
            vec_set.add(
                AlphaVector(
                    action, [self.env.getReward(s, action).value for s in State]
                )
            )
        vec_set.prune()
        return vec_set

    def __backup(self, t: int) -> VectorSet:
        """
        生成t时刻的alpha向量集合，
        alpha^a_t(s) = R(s,a) + gamma sum_{o} sum_{s'} P(s'|s,a) P(o|a,s') max_{alpha' in Gamma_{t-1}} alpha'(s')
        """
        vec_set = VectorSet(t)
        next_vec_set = self.t_to_vec_sets[t - 1]
        for action in Action:
            values = [
                self.env.getReward(state, action).value
                + self.gamma
                * sum(
                    self.env.getNextStateProb(state, action, next_state)
                    * self.env.getObsProb(action, next_state, obs)
                    * next_vec_set.max(next_state).get(next_state)
                    for obs in Observation
                    for next_state in State
                )
                for state in State
            ]
            vec_set.add(AlphaVector(action, values))
        vec_set.prune()
        return vec_set

    def plotAlphaVectors(self, t, show=True):
        """绘制t时刻的alpha向量"""
        plt.figure(figsize=(10, 6))
        # 认为老虎在左侧的概率
        tiger_left_probs = np.linspace(0, 1, 100)
        belief_list = [Belief([p, 1 - p]) for p in tiger_left_probs]
        vec_set = self.t_to_vec_sets[t]
        # 绘制alpha向量
        for alpha_vec in vec_set.alpha_vectors:
            plt.plot(
                tiger_left_probs,
                [alpha_vec.dot(belief) for belief in belief_list],
                "-.",
                label=f"{alpha_vec.action.name}",
            )
        # 绘制值函数，略微比实际值大一点，以避免遮挡alpha向量
        plt.plot(
            tiger_left_probs,
            [vec_set.getVal(belief) + 0.666 for belief in belief_list],
            "-",
            label="Value Function",
        )
        plt.title("alpha-vectors at horizon {}".format(t))
        plt.xlabel("Belief (tiger is on the left)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()
        save_path = os.path.join(Config.FIG_DIR, "alpha_vectors_{}.png".format(t))
        os.makedirs(Config.FIG_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=666)
        Log.info("Alpha-vectors at horizon {} saved to {}.".format(t, save_path))

    @staticmethod
    def demo():
        """演示alpha-vectors算法求解POMDP问题"""
        env = Tiger()
        solver = AlphaVectorSolver(env)
        solver.solve()
        for t in range(1, solver.horizon + 1):
            solver.plotAlphaVectors(t, show=False)
        # 对若干种典型信念状态进行分析，输出最优动作
        test_cases = [
            ("High LEFT", Belief([0.95, 0.05])),
            ("Uncertain", Belief([0.5, 0.5])),
            ("High RIGHT", Belief([0.05, 0.95])),
        ]
        for t in range(1, solver.horizon + 1):
            Log.info("===================== Horizon {} =====================".format(t))
            vec_set = solver.t_to_vec_sets[t]
            for name, belief in test_cases:
                best_vec = vec_set.getBestVec(belief)
                Log.info(
                    "{}: belief = {}, best action = {}, value = {:.3f}".format(
                        name, belief, best_vec.action.name, best_vec.dot(belief)
                    )
                )


if __name__ == "__main__":
    AlphaVectorSolver.demo()

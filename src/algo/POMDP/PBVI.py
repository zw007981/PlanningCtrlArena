# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from typing import List

import numpy as np

from src.map.point_finder import PointFinder
from src.util.config import Config as BaseConfig
from src.util.log import Log
from src.util.random_num_generator import RandomNumGenerator
from src.util.tiger import (
    Action,
    AlphaVector,
    Belief,
    Observation,
    Reward,
    State,
    Tiger,
)


class Config(BaseConfig):
    """PBVI算法配置"""

    GAMMA = 0.95
    """折扣系数"""
    NUM_INIT_BELIEF_PTS = 233
    """初始信念点数量"""
    BELIEF_PT_EXPANSION_THRESHOLD = 0.0666
    """信念点扩展阈值，如果一个新生成的信念点距离已有信念点的距离都大于该值，则添加到信念点集合中"""
    VAL_CONVERGE_THRESHOLD = 0.0233
    """值函数收敛阈值"""
    MAX_ITERATIONS = 200
    """最大迭代次数"""


class PBVI:
    """Point Based Value Iteration求解器"""

    def __init__(self, env: Tiger):
        """使用环境初始化"""
        self.rand_num_gen = RandomNumGenerator(888)
        """随机数生成器"""
        self.env = env
        """环境"""
        self.alpha_vector_set: List[AlphaVector] = []
        """alpha向量集合"""
        self.belief_pts: List[Belief] = []
        """信念点集合"""
        self.pt_finder = PointFinder()
        """KD tree，用于快速寻找最近的belief点"""
        self.max_val_diff = 0.0
        """两次迭代之间的最大值函数差异，可用于监测收敛情况"""
        # 初始化信念点和alpha向量集合
        for belief in self.sampleBeliefPts(Config.NUM_INIT_BELIEF_PTS):
            self.addBeliefPt(belief)
        for belief in self.belief_pts:
            self.alpha_vector_set.append(
                AlphaVector(Action.LISTEN, [Reward.LISTEN.value] * State.Size())
            )

    def solve(self) -> None:
        """求解POMDP问题"""
        Log.info("Start solving POMDP problem using PBVI algorithm...")
        for i in range(Config.MAX_ITERATIONS):
            new_alpha_vec_set = self.pointBasedValueBackup()
            if self.isConverged(new_alpha_vec_set):
                Log.info(
                    "PBVI algorithm converged after {} iterations, max value diff: {:.4f}.".format(
                        i, self.max_val_diff
                    )
                )
                break
            else:
                self.alpha_vector_set = new_alpha_vec_set
            prev_num_belief_pts = len(self.belief_pts)
            self.expandBeliefPts()
            if len(self.belief_pts) != prev_num_belief_pts:
                Log.info(
                    "Iteration {}: belief points expanded to {}.".format(
                        i,
                        len(self.belief_pts),
                    )
                )
            if i % 10 == 0:
                Log.info(
                    "Iteration {}, max value diff: {:.4f}.".format(i, self.max_val_diff)
                )
        Log.info("PBVI algorithm finished.")

    def sampleBeliefPts(self, num: int) -> List[Belief]:
        """采样num个信念点"""
        p_list = np.linspace(Config.FLOAT_EPS, 1 - Config.FLOAT_EPS, num)
        return [Belief([p, 1 - p]) for p in p_list]

    def addBeliefPt(self, belief: Belief) -> None:
        """把一个信念点添加到集合中"""
        self.belief_pts.append(belief)
        self.pt_finder.addPt(belief.state_probs[0], belief.state_probs[1])
        self.pt_finder.buildKDTree()

    def pointBasedValueBackup(self) -> List[AlphaVector]:
        """基于点的值迭代备份"""
        new_alpha_vec_set: List[AlphaVector] = []
        for belief in self.belief_pts:
            new_alpha_vec = self.backupOneBelief(belief)
            new_alpha_vec_set.append(new_alpha_vec)
        return new_alpha_vec_set

    def expandBeliefPts(self) -> None:
        """扩展信念点"""
        # 对每一个信念点进行扩展，对于每一个动作随机生成一个观测，看更新后的信念点是否足够远离其他信念点
        for belief in self.belief_pts:
            for a in Action:
                o_prime = self.rand_num_gen.getRandomEnum(Observation)
                belief_prime = self.env.updateBelief(belief, a, o_prime)
                nearest_belief = self.belief_pts[
                    self.pt_finder.findIndex(
                        belief_prime.state_probs[0], belief_prime.state_probs[1]
                    )
                ]
                if (
                    belief_prime.calL1Dist(nearest_belief)
                    > Config.BELIEF_PT_EXPANSION_THRESHOLD
                ):
                    self.addBeliefPt(belief_prime)

    def isConverged(self, new_alpha_vec_set: List[AlphaVector]) -> bool:
        """判断是否收敛"""
        self.max_val_diff = float("-inf")
        for belief in self.belief_pts:
            old_val = max(alpha.dot(belief) for alpha in self.alpha_vector_set)
            new_val = max(alpha.dot(belief) for alpha in new_alpha_vec_set)
            diff = abs(old_val - new_val)
            self.max_val_diff = max(self.max_val_diff, diff)
            if diff > Config.VAL_CONVERGE_THRESHOLD:
                # 如果值函数差异超过阈值，则认为不收敛
                return False
        return True

    def calObsProbGivenBeliefAction(
        self, o_prime: Observation, a: Action, b: Belief
    ) -> float:
        """计算上一步处于信念b，采取动作a，获得观测o'的概率"""
        p = 0.0
        for s in State:
            for s_prime in State:
                p += (
                    self.env.getObsProb(a, s_prime, o_prime)
                    * self.env.getNextStateProb(s, a, s_prime)
                    * b.getProb(s)
                )
        return p

    def backupOneBelief(self, belief: Belief) -> AlphaVector:
        """为信念点生成新的alpha-向量"""
        best_alpha_vec = None
        best_val = float("-inf")
        for a in Action:
            q_values = []
            for s in State:
                # alpha^a(s) = r(s,a) + gamma * sum_o' V(b') * P(o'|b,a)
                # = r(s,a) + gamma * sum_{o'} V(b') * [sum_{s',s} P(o'|s,a,s') * P(s'|s,a) * b(s)]
                immediate_reward = self.env.getReward(s, a).value
                future_val = 0.0
                for o_prime in Observation:
                    b_prime = self.env.updateBelief(belief, a, o_prime)
                    v_b_prime = max(
                        alpha.dot(b_prime) for alpha in self.alpha_vector_set
                    )
                    future_val += v_b_prime * self.calObsProbGivenBeliefAction(
                        o_prime, a, belief
                    )
                # 计算得到处于状态s，执行动作a获得的期望回报值；后续取最大的动作即可获得值函数
                q_values.append(immediate_reward + Config.GAMMA * future_val)
            new_alpha_vec = AlphaVector(a, q_values)
            val = new_alpha_vec.dot(belief)
            if val > best_val + Config.FLOAT_EPS:
                best_alpha_vec = new_alpha_vec
                best_val = val
        if best_alpha_vec is None:
            Log.error("No alpha vector found!!!")
            raise ValueError("No alpha vector found!!!")
        return best_alpha_vec

    @staticmethod
    def Demo():
        """PBVI算法演示"""
        env = Tiger()
        solver = PBVI(env)
        solver.solve()
        test_cases = [
            ("High LEFT", Belief([0.95, 0.05])),
            ("Uncertain", Belief([0.5, 0.5])),
            ("High RIGHT", Belief([0.05, 0.95])),
        ]
        for name, belief in test_cases:
            best_alpha_vec = max(solver.alpha_vector_set, key=lambda x: x.dot(belief))
            Log.info(
                "Case {}: belief = {}, best action = {}.".format(
                    name,
                    belief,
                    best_alpha_vec.action.name,
                )
            )


if __name__ == "__main__":
    PBVI.Demo()

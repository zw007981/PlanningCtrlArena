# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from src.util.config import Config
from src.util.log import Log
from src.util.random_num_generator import RandomNumGenerator


class State(Enum):
    """状态空间"""

    LEFT = 0
    """老虎在左"""
    RIGHT = 1
    """老虎在右"""

    @staticmethod
    def Size() -> int:
        """获取状态空间的大小"""
        return len(State)


class Action(Enum):
    """动作空间"""

    OPEN_LEFT = 0
    """开左门"""
    OPEN_RIGHT = 1
    """开右门"""
    LISTEN = 2
    """倾听"""

    @staticmethod
    def Size() -> int:
        """获取动作空间的大小"""
        return len(Action)

    @staticmethod
    def IsTigerBehindDoor(state: State, action: "Action") -> bool:
        """判断老虎是否在门后"""
        return (action == Action.OPEN_LEFT and state == State.LEFT) or (
            action == Action.OPEN_RIGHT and state == State.RIGHT
        )


class Reward(Enum):
    """奖励空间"""

    TIGER = -100.0
    """老虎在门后"""
    TREASURE = 10.0
    """宝藏在门后"""
    LISTEN = -1.0
    """倾听的奖励"""

    @staticmethod
    def Size() -> int:
        """获取奖励空间的大小"""
        return len(Reward)

    @staticmethod
    def MaxVal() -> float:
        """获取奖励空间的最大值"""
        return max([r.value for r in Reward])

    @staticmethod
    def MinVal() -> float:
        """获取奖励空间的最小值"""
        return min([r.value for r in Reward])


class Observation(Enum):
    """观测空间"""

    LEFT = 0
    """观测到老虎在左"""
    RIGHT = 1
    """观测到老虎在右"""

    @staticmethod
    def Size() -> int:
        """获取观测空间的大小"""
        return len(Observation)

    @staticmethod
    def IsObsCorrect(state: State, obs: "Observation") -> bool:
        """判断观测是否正确"""
        return (obs == Observation.LEFT and state == State.LEFT) or (
            obs == Observation.RIGHT and state == State.RIGHT
        )


class Belief:
    """信念状态"""

    def __init__(self, state_probs: NDArray | List[float]):
        """使用各个状态的概率分布初始化信念状态，会自动执行归一化操作"""
        if isinstance(state_probs, list):
            state_probs = np.array(state_probs)
        self.state_probs: NDArray = state_probs
        """信念状态，存储为各个状态的概率分布"""
        if self.state_probs.shape[0] != State.Size():
            Log.error(
                "The size of state_probs {} does not match the size of state space {}!!!".format(
                    self.state_probs.shape[0], State.Size()
                )
            )
            raise ValueError(
                "The size of state_probs does not match the size of state space!!!"
            )
        self.__normalize()

    def __repr__(self) -> str:
        str_to_return = "{"
        for i in range(State.Size()):
            str_to_return += "P_{}: {:.3f}".format(State(i).name, self.state_probs[i])
            if i < State.Size() - 1:
                str_to_return += ","
        str_to_return += "}"
        return str_to_return

    def __normalize(self) -> None:
        """将信念状态归一化"""
        sum_ = self.state_probs.sum()
        if abs(sum_) < Config.ZERO_EPS:
            # 有些时候会出现agent不可能进入这种状态的情况，如在执行开门指令后获得老虎在左边的观测，
            # 简单的指定每种状态的概率相等即可，因为这种信念对应的概率权重为0无关紧要
            self.state_probs = np.ones(State.Size()) / State.Size()
        else:
            self.state_probs /= sum_

    def getProb(self, state: State) -> float:
        """获取状态state的概率"""
        return self.state_probs[state.value]

    def calL1Dist(self, rhs: "Belief") -> float:
        """计算两个信念状态之间的L1距离"""
        return np.abs(self.state_probs - rhs.state_probs).sum()


class AlphaVector:
    """alpha-向量类，表示在给定第一个动作后各个状态的期望回报值"""

    def __init__(self, action: Action, values: NDArray | List[float]):
        self.action = action
        """第一个动作"""
        if isinstance(values, list):
            values = np.array(values, dtype=np.float64)
        self.values = values
        """各个状态对应的期望回报值"""
        if len(values) != State.Size():
            Log.error(
                "The length of values ({}) does not match the size of state space ({})!!!".format(
                    len(values), State.Size()
                )
            )
            raise ValueError("Values length does not match state space size!!!")

    def __repr__(self):
        return "AlphaVector({}, {})".format(self.action.name, self.values.round(2))

    def get(self, state: State) -> float:
        """获取状态对应的期望回报值"""
        return self.values[state.value]

    def dot(self, belief: Belief) -> float:
        """输入信念状态，计算Q值：Q = sum_{s} alpha(s) b(s)"""
        return np.dot(belief.state_probs, self.values)

    def isDominated(self, rhs: "AlphaVector") -> bool:
        """判断当前向量是否被rhs支配"""
        is_dominated = True
        for i in range(State.Size()):
            if self.values[i] > rhs.values[i] + Config.FLOAT_EPS:
                is_dominated = False
                break
        return is_dominated


class Tiger:
    """Tiger问题的环境类"""

    P_OBS_CORRECT = 0.85
    """观测到正确状态的概率"""

    def __init__(self):
        self.rand_num_gen = RandomNumGenerator(233)
        """随机数生成器"""
        self.state = self.rand_num_gen.getRandomEnum(State)
        """当前状态"""
        self.state_trans_model: Dict[State, Dict[Action, Dict[State, float]]] = {}
        """状态转移模型，存储为从状态到动作再到下一个状态的概率分布"""
        self.__constructStateTransModel()
        self.reward_func: Dict[State, Dict[Action, Reward]] = {}
        """奖励函数，存储为从状态到动作再到奖励的映射"""
        self.__constructRewardFunc()
        self.obs_model: Dict[Action, Dict[State, Dict[Observation, float]]] = {}
        """观测模型，存储为从上一步的动作到这一步的状态再到观测的概率分布"""
        self.__constructObsModel()

    @staticmethod
    def Insert(dict_to_insert: Dict, keys: List[Any], value: Any):
        """向字典中插入键值对，如果中间的键不存在，则创建对应的字典"""
        if len(keys) == 0:
            return
        elif len(keys) == 1:
            dict_to_insert[keys[0]] = value
        else:
            cur_dict = dict_to_insert
            for key in keys[:-1]:
                if key not in cur_dict:
                    cur_dict[key] = {}
                cur_dict = cur_dict[key]
            cur_dict[keys[-1]] = value

    @staticmethod
    def PlotHistory(
        state_history: List[State],
        action_history: List[Action],
        belief_history: List[Belief] = [],
        title: str = "",
    ) -> None:
        """绘制Tiger问题的历史记录"""
        if len(belief_history) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        if title != "":
            fig.suptitle(title)
        else:
            fig.suptitle("Tiger Problem History")
        # 绘制第一幅图，状态-动作图
        ax = axes[0] if len(belief_history) > 0 else axes  # type: ignore
        ax.plot(
            range(len(state_history)), [s.value for s in state_history], label="State"
        )
        ax.plot(
            range(len(action_history)),
            [0.5 if a == Action.LISTEN else a.value for a in action_history],
            "-o",
            label="Action",
        )
        ax.grid(True, linestyle="--")
        ax.set_xlabel("Time Step")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks([0, 0.5, 1], ["LEFT", "LISTEN", "RIGHT"])
        ax.legend()
        # 绘制第二幅图，信念状态图
        if len(belief_history) > 0:
            ax = axes[1]  # type: ignore
            for state in State:
                ax.plot(
                    range(len(belief_history)),
                    [b.getProb(state) for b in belief_history],
                    "-.",
                    label=state.name,
                )
            ax.grid(True, linestyle="--")
            ax.set_xlabel("Time Step")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel("Probability")
            ax.legend()
        plt.tight_layout()
        plt.show()
        fig_path = os.path.join(Config.FIG_DIR, "tiger.png")
        if title != "":
            fig_path = os.path.join(Config.FIG_DIR, "{}.png".format(title))
        fig.savefig(fig_path, dpi=233)
        Log.info("Figure saved to {}.".format(fig_path))

    def getNextStateProb(
        self, state: State, action: Action, next_state: State
    ) -> float:
        """获取在state状态下执行action后转移到next_state的概率"""
        return self.state_trans_model[state][action][next_state]

    def getReward(self, state: State, action: Action) -> Reward:
        """获取在state状态下执行action后的奖励"""
        return self.reward_func[state][action]

    def getObsProb(self, pre_action: Action, state: State, obs: Observation) -> float:
        """获取执行pre_action后在state状态下观测到obs的概率"""
        return self.obs_model[pre_action][state][obs]

    def reset(self) -> None:
        """重置环境"""
        self.state = self.rand_num_gen.getRandomEnum(State)

    def updateBelief(self, b: Belief, a: Action, o_prime: Observation) -> Belief:
        """上一步处于信念b，采取动作a，获得观测o'，更新信念状态"""
        # 这里的计算方法与原始公式不同，没有计算分母而是在计算出所有的分子后统一执行归一化
        # 分子：b'(s') = Z(s',a,o') * sum_s T(s,a,s') * b(s)
        s_prime_probs = np.zeros(State.Size())
        for s_prime in State:
            z = self.getObsProb(a, s_prime, o_prime)
            s_prime_probs[s_prime.value] = z * sum(
                self.getNextStateProb(s, a, s_prime) * b.getProb(s) for s in State
            )
        return Belief(s_prime_probs)

    def sampleNextStateAndObs(
        self, state: State, action: Action, phi: float = -1.0
    ) -> Tuple[State, Observation]:
        """
        根据当前状态和动作随机采样下一个状态和观测，为了兼容DESPOT中的
        deterministic simulative model，也可以输入一个参数phi来获取确定性的结果。
        """
        if phi < -Config.ZERO_EPS:
            phi = self.rand_num_gen.getRandomFloat()
        if action == Action.LISTEN:
            # 倾听不改变状态
            next_state = deepcopy(state)
            if phi < Tiger.P_OBS_CORRECT:
                # 获得正确的观测
                obs = Observation.LEFT if state == State.LEFT else Observation.RIGHT
            else:
                obs = Observation.RIGHT if state == State.LEFT else Observation.LEFT
            return next_state, obs
        else:
            # 开门会重置状态并随机选择一个观测
            if phi <= 0.25:
                next_state = State.LEFT
                obs = Observation.LEFT
            elif phi <= 0.5:
                next_state = State.LEFT
                obs = Observation.RIGHT
            elif phi <= 0.75:
                next_state = State.RIGHT
                obs = Observation.RIGHT
            else:
                next_state = State.RIGHT
                obs = Observation.LEFT
            return next_state, obs

    def step(self, action: Action) -> Tuple[Reward, Observation, bool]:
        """
        执行动作action，返回奖励、观测和是否结束。
        如果选择打开门则直接结束，否则不结束。
        """
        next_state, obs = self.sampleNextStateAndObs(self.state, action)
        if action == Action.LISTEN:
            return Reward.LISTEN, obs, False
        else:
            reward = (
                Reward.TIGER
                if Action.IsTigerBehindDoor(self.state, action)
                else Reward.TREASURE
            )
            self.state = next_state
            return reward, obs, True

    def __constructStateTransModel(self) -> None:
        """构造状态转移模型"""
        for state in State:
            for next_state in State:
                # 开门后重置状态（老虎随机分布）
                for action in [Action.OPEN_LEFT, Action.OPEN_RIGHT]:
                    Tiger.Insert(
                        self.state_trans_model, [state, action, next_state], 0.5
                    )
                # 倾听不改变状态
                if state == next_state:
                    Tiger.Insert(
                        self.state_trans_model, [state, Action.LISTEN, next_state], 1.0
                    )
                else:
                    Tiger.Insert(
                        self.state_trans_model, [state, Action.LISTEN, next_state], 0.0
                    )

    def __constructRewardFunc(self) -> None:
        """构造奖励函数"""
        for state in State:
            for action in [Action.OPEN_LEFT, Action.OPEN_RIGHT]:
                if Action.IsTigerBehindDoor(state, action):
                    Tiger.Insert(self.reward_func, [state, action], Reward.TIGER)
                else:
                    Tiger.Insert(self.reward_func, [state, action], Reward.TREASURE)
            Tiger.Insert(self.reward_func, [state, Action.LISTEN], Reward.LISTEN)

    def __constructObsModel(self) -> None:
        """构造观测模型"""
        # 选择倾听
        action = Action.LISTEN
        for state in State:
            for obs in Observation:
                if Observation.IsObsCorrect(state, obs):
                    # 正确的观测概率为P_OBS_CORRECT
                    Tiger.Insert(
                        self.obs_model, [action, state, obs], Tiger.P_OBS_CORRECT
                    )
                else:
                    # 错误的观测概率为1 - P_OBS_CORRECT
                    Tiger.Insert(
                        self.obs_model, [action, state, obs], 1 - Tiger.P_OBS_CORRECT
                    )
        # 选择开门，随机选择一个观测即可
        for action in [Action.OPEN_LEFT, Action.OPEN_RIGHT]:
            for state in State:
                for obs in Observation:
                    Tiger.Insert(
                        self.obs_model, [action, state, obs], 1.0 / Observation.Size()
                    )


class ParticleFilter:
    """无权重粒子滤波器 (Unweighted Particle Filter)"""

    NOISE_PARTICLE_RATIO = 0.0888
    """噪声比例，这个比例的粒子会被均匀分配到每个状态"""
    RAND_NUM_GEN = RandomNumGenerator(233)
    """随机数生成器"""

    def __init__(
        self, env: Tiger, num_particles: int, enable_noise: bool = True
    ) -> None:
        """使用所在环境、粒子数量和是否添加噪声初始化粒子滤波器"""
        self.env = env
        """所处环境"""
        self.num_particles = num_particles
        """粒子数量"""
        self.enable_noise = enable_noise
        """是否添加噪声"""
        # 常规的存储方法是存储一个长度为num_particles的数组，数组中的每个元素都是一个状态
        # 但是这样会占用大量内存，我们把相同状态的粒子放在一起并按照State进行排序从而生成
        # 一个列表，借助这个列表为每个粒子分配一个序号。必要时，使用粒子序号进行查找。
        # 可进一步精简为一个(State.Size() - 1) * 1的数组，但是考虑到可读性暂时先这样。
        self.state_index_ranges: NDArray = np.zeros((State.Size(), 2))
        """从状态到粒子序号范围的映射，形状为State.Size() * 2，第一列为最小序号，第二列为最大序号"""
        num_particles_per_state = float(num_particles) / State.Size()
        self.__constructStateRange(np.full(State.Size(), num_particles_per_state))

    def __constructStateRange(self, state_to_num: NDArray) -> None:
        """根据从状态到粒子数量的映射，构造从状态到粒子序号范围的映射"""
        accumulated = 0.0
        for state in State:
            # min_index
            self.state_index_ranges[state.value, 0] = accumulated
            accumulated += state_to_num[state.value]
            # max_index
            self.state_index_ranges[state.value, 1] = accumulated

    def __getStateFromIndex(self, particle_index: int) -> State:
        """根据粒子序号获取对应的状态"""
        if particle_index < 0 or particle_index > self.num_particles:
            Log.info(
                "Particle index {} is out of range {}-{}!!!".format(
                    particle_index, 0, self.num_particles - 1
                )
            )
            raise ValueError("Particle index is out of range!!!")
        in_range = (self.state_index_ranges[:, 0] <= particle_index) & (
            particle_index < self.state_index_ranges[:, 1]
        )
        state_idx = np.where(in_range)[0]
        if state_idx.size == 0:
            return State.RIGHT
        return State(state_idx[0])

    def getBelief(self) -> Belief:
        """获取对应信念状态"""
        state_probs = self.state_index_ranges[:, 1] - self.state_index_ranges[:, 0]
        # 会自动进行归一化操作
        return Belief(state_probs)

    def sample(self) -> State:
        """从现有分布中采样一个状态"""
        particle_index = ParticleFilter.RAND_NUM_GEN.getRandomNum(
            0, self.num_particles - 1
        )
        return self.__getStateFromIndex(particle_index)

    def update(self, action: Action, obs: Observation) -> None:
        """在实际环境中执行了动作action获得了观测obs后进行更新"""
        state_to_num = np.zeros(State.Size())
        # 当前已经获得的新粒子的数量和需要填充的粒子数量
        total_num = 0
        num_particles_to_fill = self.num_particles
        if self.enable_noise:
            num_particles_to_fill *= 1.0 - ParticleFilter.NOISE_PARTICLE_RATIO
        # 设置最大尝试次数，避免死循环
        for _ in range(666 * self.num_particles):
            state = self.sample()
            next_state, next_obs = self.env.sampleNextStateAndObs(state, action)
            if next_obs == obs:
                state_to_num[next_state.value] += 1.0
                total_num += 1
            if total_num >= num_particles_to_fill:
                break
        if total_num < self.num_particles:
            residual_per_state = float(self.num_particles - total_num) / State.Size()
            state_to_num += residual_per_state
        self.__constructStateRange(state_to_num)


class History:
    """历史记录"""

    def __init__(self, action_vals: NDArray, obs_vals: NDArray) -> None:
        self.action_vals = action_vals
        """历史动作值"""
        self.obs_vals = obs_vals
        """历史观察值"""
        self.hash_val = self.__calHashVal()
        """哈希值"""

    @staticmethod
    def ConstructFromList(
        action_list: List[Action], obs_list: List[Observation]
    ) -> "History":
        """从动作列表和观察列表构造历史记录"""
        action_vals = np.array([action.value for action in action_list])
        obs_vals = np.array([obs.value for obs in obs_list])
        return History(action_vals, obs_vals)

    @property
    def empty(self) -> bool:
        """是否为空"""
        return self.action_vals.size == 0 and self.obs_vals.size == 0

    @property
    def action(self) -> Action:
        """最后一个动作"""
        if self.action_vals.size == 0:
            Log.error("Action history is empty!")
            raise ValueError("Action history is empty!")
        return Action(self.action_vals[-1])

    @property
    def obs(self) -> Observation:
        """最后一个观察"""
        if self.obs_vals.size == 0:
            Log.error("Observation history is empty!")
            raise ValueError("Observation history is empty!")
        return Observation(self.obs_vals[-1])

    def __hash__(self) -> int:
        return self.hash_val

    def __eq__(self, rhs: object) -> bool:
        if not isinstance(rhs, History):
            return False
        return np.array_equal(self.action_vals, rhs.action_vals) and np.array_equal(
            self.obs_vals, rhs.obs_vals
        )

    def __repr__(self) -> str:
        action_names = [Action(val).name for val in self.action_vals]
        obs_names = [Observation(val).name for val in self.obs_vals]
        return "History[action_vals={}, obs_vals={}]".format(action_names, obs_names)

    def __calHashVal(self) -> int:
        """自定义哈希值计算方法"""
        # 每个元素乘以一个递增的权重
        action_hash = 0
        if self.action_vals.size > 0:
            weights_action = np.arange(1, self.action_vals.size + 1, dtype=np.int64)
            action_hash = np.dot(self.action_vals.astype(np.int64), weights_action)
        obs_hash = 0
        if self.obs_vals.size > 0:
            weights_obs = np.arange(1, self.obs_vals.size + 1, dtype=np.int64)
            obs_hash = np.dot(self.obs_vals.astype(np.int64), weights_obs)
        return int(
            action_hash
            + obs_hash * 31
            + (self.action_vals.size + self.obs_vals.size) * 17
        )

    def augment(
        self, new_action: Optional[Action] = None, new_obs: Optional[Observation] = None
    ) -> "History":
        """在当前历史记录上增加一个动作和/或观察，生成新的历史记录"""
        if new_action is None and new_obs is None:
            Log.error("Both action and observation are None!!!")
            raise ValueError("Both action and observation are None!!!")
        new_action_vals = np.copy(self.action_vals)
        new_obs_vals = np.copy(self.obs_vals)
        if new_action is not None:
            new_action_vals = np.append(new_action_vals, new_action.value)
        if new_obs is not None:
            new_obs_vals = np.append(new_obs_vals, new_obs.value)
        return History(new_action_vals, new_obs_vals)


if __name__ == "__main__":
    env = Tiger()

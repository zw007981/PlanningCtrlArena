# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import pytest

from src.util.config import Config
from src.util.tiger import Action, Observation, Reward, State, Tiger


@pytest.fixture
def tiger_env():
    """
    创建Tiger环境fixture，供后续测试使用
    """
    return Tiger()


def test_initial_state(tiger_env):
    """
    测试Tiger环境初始化时的基本属性
    """
    assert tiger_env.state in [State.LEFT, State.RIGHT]


def test_step_listen(tiger_env):
    """
    测试在倾听(LISTEN)动作下的step函数返回
    """
    old_state = tiger_env.state
    reward, obs, done = tiger_env.step(Action.LISTEN)
    # 倾听动作的奖励应为R_LISTEN
    assert reward == Reward.LISTEN
    # 由于是LISTEN，不会结束
    assert done is False
    # 观测应该非NONE
    assert obs in [
        Observation.LEFT,
        Observation.RIGHT,
    ]
    # 倾听不改变状态
    assert tiger_env.state == old_state


def test_step_open_left_when_state_left(tiger_env):
    """
    测试打开左门(OPEN_LEFT)时，如果老虎在左边，奖励应为R_TIGER，否则为R_TREASURE
    这里为了可测性，先手动修改状态为LEFT
    """
    tiger_env.state = State.LEFT
    reward, obs, done = tiger_env.step(Action.OPEN_LEFT)
    # 打开门后应结束回合
    assert done is True
    # 老虎在左边时，打开左门应得到 R_TIGER
    assert reward == Reward.TIGER
    assert obs in [
        Observation.LEFT,
        Observation.RIGHT,
    ]


def test_step_open_left_when_state_right(tiger_env):
    """
    测试打开左门(OPEN_LEFT)时，如果老虎在右边，奖励应为R_TREASURE
    这里为了可测性，先手动修改状态为RIGHT
    """
    tiger_env.state = State.RIGHT
    reward, obs, done = tiger_env.step(Action.OPEN_LEFT)
    # 打开门后应结束回合
    assert done is True
    # 老虎在右边时，打开左门应得到 R_TREASURE
    assert reward == Reward.TREASURE
    assert obs in [
        Observation.LEFT,
        Observation.RIGHT,
    ]


def test_env_resets_after_open(tiger_env):
    """
    测试进行OPEN_LEFT或OPEN_RIGHT动作后，环境是否正确重置了状态
    """
    # 打开门后，done=True，并会重置state
    _, _, done = tiger_env.step(Action.OPEN_LEFT)
    assert done, "打开门后回合应该结束"
    # 再检查一下新的state是否和旧的不一定相同，但应该在左右之间随机选择
    assert tiger_env.state in [
        State.LEFT,
        State.RIGHT,
    ]


def test_state_transition_listen_action(tiger_env):
    """
    测试LISTEN动作的状态转移模型:
    - 在当前状态执行LISTEN时，应该100%保持在相同状态
    - 转移到其他状态的概率应为0
    """
    for current_state in State:
        # 同一状态转移概率应为1.0
        pytest.approx(
            tiger_env.getNextStateProb(current_state, Action.LISTEN, current_state), 1.0
        )
        # 其他状态转移概率应为0.0
        for next_state in State:
            if next_state != current_state:
                pytest.approx(
                    tiger_env.getNextStateProb(
                        current_state, Action.LISTEN, next_state
                    ),
                    0.0,
                )


def test_state_transition_open_actions(tiger_env):
    """
    测试开门动作(OPEN_LEFT, OPEN_RIGHT)的状态转移模型:
    - 开门后应该有50%概率转移到LEFT，50%概率转移到RIGHT
    """
    for action in [Action.OPEN_LEFT, Action.OPEN_RIGHT]:
        for current_state in State:
            pytest.approx(
                tiger_env.getNextStateProb(current_state, action, State.LEFT), 0.5
            )
            pytest.approx(
                tiger_env.getNextStateProb(current_state, action, State.RIGHT), 0.5
            )


def test_obs_model_listen_action_details(tiger_env):
    """
    详细测试倾听动作的观测模型:
    - 在状态LEFT时，应有P_OBS_CORRECT(0.85)概率观测到OBS_LEFT
    - 在状态LEFT时，应有(1-P_OBS_CORRECT)(0.15)概率观测到OBS_RIGHT
    - 在状态RIGHT时，应有P_OBS_CORRECT(0.85)概率观测到OBS_RIGHT
    - 在状态RIGHT时，应有(1-P_OBS_CORRECT)(0.15)概率观测到OBS_LEFT
    """
    # 测试倾听时的观测模型
    action = Action.LISTEN
    # 状态为LEFT的情况
    pytest.approx(
        tiger_env.getObsProb(action, State.LEFT, Observation.LEFT),
        Tiger.P_OBS_CORRECT,
    )
    pytest.approx(
        tiger_env.getObsProb(action, State.LEFT, Observation.RIGHT),
        1 - Tiger.P_OBS_CORRECT,
    )
    # 状态为RIGHT的情况
    pytest.approx(
        tiger_env.getObsProb(action, State.RIGHT, Observation.RIGHT),
        Tiger.P_OBS_CORRECT,
    )
    pytest.approx(
        tiger_env.getObsProb(action, State.RIGHT, Observation.LEFT),
        1 - Tiger.P_OBS_CORRECT,
    )


def test_obs_model_open_actions_details(tiger_env):
    """
    详细测试开门动作的观测模型:
    - 执行OPEN_LEFT或OPEN_RIGHT时，无论当前状态如何，应该随机返回一个观测
    """
    for action in [Action.OPEN_LEFT, Action.OPEN_RIGHT]:
        for state in State:
            p = 1.0 / Observation.Size()
            pytest.approx(tiger_env.getObsProb(action, state, Observation.LEFT), p)
            pytest.approx(tiger_env.getObsProb(action, state, Observation.RIGHT), p)


def test_obs_model_probability_sum(tiger_env):
    """
    测试观测模型中：对于某个给定的状态和动作，所有观测的概率之和应为1
    """
    for action in Action:
        for state in State:
            total_prob = sum(
                tiger_env.getObsProb(action, state, obs) for obs in Observation
            )
            # 由于浮点数精度问题，可使用approx
            assert abs(total_prob - 1.0) < Config.FLOAT_EPS


def test_state_transition_probability_sum(tiger_env):
    """
    测试状态转移模型中：对于某个给定的状态和动作，所有下一状态的概率之和应为1
    """
    for state in State:
        for action in Action:
            total_prob = sum(
                tiger_env.getNextStateProb(state, action, next_state)
                for next_state in State
            )
            assert abs(total_prob - 1.0) < Config.FLOAT_EPS


def test_get_reward_function(tiger_env):
    """
    测试奖励函数：在不同状态下执行各动作的奖励是否正确
    """
    # 如果老虎在LEFT，打开左门奖励=R_TIGER，否则R_TREASURE
    assert tiger_env.getReward(State.LEFT, Action.OPEN_LEFT) == Reward.TIGER
    assert tiger_env.getReward(State.LEFT, Action.OPEN_RIGHT) == Reward.TREASURE
    # 如果老虎在RIGHT，打开右门奖励=R_TIGER，否则R_TREASURE
    assert tiger_env.getReward(State.RIGHT, Action.OPEN_LEFT) == Reward.TREASURE
    assert tiger_env.getReward(State.RIGHT, Action.OPEN_RIGHT) == Reward.TIGER
    # 倾听动作奖励=R_LISTEN
    for s in [State.LEFT, State.RIGHT]:
        assert tiger_env.getReward(s, Action.LISTEN) == Reward.LISTEN

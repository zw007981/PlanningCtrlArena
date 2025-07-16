# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())

import random

import pytest

from src.algo.POMDP.POMCP import ParticleFilter
from src.util.tiger import Action, Observation, State, Tiger


@pytest.fixture
def particle_filter() -> ParticleFilter:
    env = Tiger()
    particle_filter = ParticleFilter(env, 1000)
    return particle_filter


def test_particle_filter_init(particle_filter: ParticleFilter) -> None:
    """粒子滤波器初始化后各状态对应的概率应该相同"""
    belief = particle_filter.getBelief()
    pytest.approx(belief.getProb(State.LEFT), 0.5)
    pytest.approx(belief.getProb(State.RIGHT), 0.5)


def test_obs_tiger_on_left(particle_filter: ParticleFilter) -> None:
    """连续多次观察到老虎在左边，粒子滤波器的状态分布应该向左偏移"""
    belief = particle_filter.getBelief()
    p_left = belief.getProb(State.LEFT)
    action = Action.LISTEN
    for _ in range(3):
        obs = Observation.LEFT
        particle_filter.update(action, obs)
        belief = particle_filter.getBelief()
        assert p_left <= belief.getProb(State.LEFT)
        p_left = belief.getProb(State.LEFT)


def test_obs_tiger_on_left_or_right(particle_filter: ParticleFilter) -> None:
    """观察到老虎处于左边或右边，粒子滤波器的状态分布应该向对应的方向偏移"""
    belief = particle_filter.getBelief()
    p_left = belief.getProb(State.LEFT)
    action = Action.LISTEN
    # 第一次观察到在左边
    particle_filter.update(action, Observation.LEFT)
    belief = particle_filter.getBelief()
    assert p_left <= belief.getProb(State.LEFT)
    p_left = belief.getProb(State.LEFT)
    # 第二次观察到在右边
    particle_filter.update(action, Observation.RIGHT)
    belief = particle_filter.getBelief()
    assert p_left >= belief.getProb(State.LEFT)
    p_left = belief.getProb(State.LEFT)
    # 第三次观察到在左边
    particle_filter.update(action, Observation.LEFT)
    belief = particle_filter.getBelief()
    assert p_left <= belief.getProb(State.LEFT)


def test_open_the_door(particle_filter: ParticleFilter) -> None:
    """连续多次打开门，由于开门后重置状态，粒子滤波器应该接近均匀分布"""
    belief = particle_filter.getBelief()
    for _ in range(3):
        action = random.choice([Action.OPEN_LEFT, Action.OPEN_RIGHT])
        obs = random.choice([Observation.LEFT, Observation.RIGHT])
        particle_filter.update(action, obs)
        belief = particle_filter.getBelief()
        p_left = belief.getProb(State.LEFT)
        pytest.approx(p_left, 0.5, abs=0.05)

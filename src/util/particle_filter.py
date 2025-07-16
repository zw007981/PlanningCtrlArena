import matplotlib.pyplot as plt
import numpy as np

# 假设我们有一个不准确的温度计：真实温度：temp(t) = 25 + 3*sin(t/2)的自然波动
# 基于粒子滤波对温度进行估计。
true_temp = lambda t: 25 + 3 * np.sin(t / 2)  # 真实温度函数
obs_noise = 5.0  # 观测噪声幅度
process_noise = 0.5  # 过程噪声幅度
n_particles = 1000  # 粒子数量
timesteps = 50  # 时间步长


class ParticleFilter:
    def __init__(self):
        self.particles = np.linspace(20, 30, n_particles)  # 初始粒子均匀分布
        self.weights = np.ones(n_particles) / n_particles

    def predict(self, t):
        # 过程模型：添加过程噪声和温度自然变化
        delta = true_temp(t) - true_temp(t - 1)  # 真实温度变化量
        self.particles += delta + np.random.normal(0, process_noise, n_particles)

    def update(self, measurement):
        # 观测更新：计算权重（与观测值的相似度）
        dist = np.abs(self.particles - measurement)
        self.weights = 1 / (1 + dist**2)  # 逆距离平方加权
        self.weights /= np.sum(self.weights)  # 归一化

    def resample(self):
        # 系统重采样避免粒子退化
        indices = np.random.choice(range(n_particles), size=n_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(n_particles) / n_particles


# 初始化
pf = ParticleFilter()
true_states = []
estimates = []
observations = []

# 主循环
for t in range(1, timesteps + 1):
    # 真实状态演进
    true = true_temp(t)
    true_states.append(true)

    # 生成带噪声的观测
    obs = true + np.random.uniform(-obs_noise, obs_noise)
    observations.append(obs)

    # 粒子滤波步骤
    pf.predict(t)
    pf.update(obs)
    pf.resample()

    # 状态估计（加权平均）
    estimates.append(np.mean(pf.particles))

plt.figure(figsize=(12, 6))
plt.plot(true_states, "g-", label="True Temperature")
plt.plot(observations, "r.", alpha=0.5, label="Observations")
plt.plot(estimates, "b--", label="PF Estimate")
plt.legend()
plt.title("Particle Filter Tracking Performance")
plt.xlabel("Time")
plt.ylabel("Temperature(℃)")
plt.show()

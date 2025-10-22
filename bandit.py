
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10): #初始化10个老虎臂
        self.rates = np.random.rand(arms) #一旦设置，就固定不变

    def play(self, arm):
        rate = self.rates[arm] #根据老虎臂的索引，获取对应老虎臂的奖励概率
        if rate > np.random.rand(): #根据老虎臂的奖励概率，判断是否奖励1，等价于：以概率 rate 返回 1，以概率 1−p 返回 0
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon #探索率，即以概率 epsilon 随机选择一个动作，以概率 1−epsilon 选择当前最优动作
        self.Qs = np.zeros(action_size) #记录每个动作的价值估计，初始值为0
        self.ns = np.zeros(action_size) #记录每个动作被选择的次数，初始值为0

    def update(self, action, reward):
        self.ns[action] += 1 #更新动作 action 被选择的次数
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action] #更新动作 action 的价值估计

    def get_action(self):
        if np.random.rand() < self.epsilon: #以概率 epsilon 随机选择一个动作
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs) #以概率 1−epsilon 选择当前最优动作


if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print(total_reward)

    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()

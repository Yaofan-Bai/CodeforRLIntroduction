import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    """非平稳10臂赌博机环境"""
    def __init__(self, k=10):
        self.k = k
        # 初始时所有动作的真实价值相等（例如都设为0），或者随机初始化
        self.true_values = np.random.randn(k)
        self.optimal_action = np.argmax(self.true_values)

    def step(self):
        # 核心：模拟非平稳性。每一时刻给所有动作的真实价值加上一个微小的随机偏置
        # 例如：均值为0，标准差为0.01的正态分布噪声
        self.true_values += np.random.normal(0, 0.01, self.k)
        self.optimal_action = np.argmax(self.true_values)

    def get_reward(self, action):
        # 奖励 = 当前真实的期望 + 噪声
        return self.true_values[action] + np.random.randn()

class Agent:
    """增加步长选择的 Agent"""
    def __init__(self, k=10, epsilon=0.1, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha  # 如果 alpha 为 None，则使用 1/n (样本平均)
        self.q_estimates = np.zeros(k)
        self.action_counts = np.zeros(k)

    def select_action(self, method='epsilon_greedy'):
        if method == 'epsilon_greedy':
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.k)
            return np.argmax(self.q_estimates)
        elif method == 'ucb':
            # UCB 策略
            ucb_values = self.q_estimates + np.sqrt(2 * np.log(np.sum(self.action_counts) + 1) / (self.action_counts + 1e-5))
            return np.argmax(ucb_values)
        elif method == 'gradient':
            # 梯度上升策略（假设偏好值初始化为0）
            preferences = self.q_estimates
            exp_preferences = np.exp(preferences - np.max(preferences)) # 数值稳定性, 减去最大值，防止溢出，但是保证逻辑相通
            action_probs = exp_preferences / np.sum(exp_preferences)
            return np.random.choice(self.k, p=action_probs)

    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        
        # 核心逻辑：判断是使用“样本平均”还是“固定步长”
        if self.alpha is None:
            step_size = 1.0 / self.action_counts[action] # 样本平均 (适合平稳)
        else:
            step_size = self.alpha # 固定步长 (适合非平稳)
            
        self.q_estimates[action] += step_size * (reward - self.q_estimates[action])

def run_non_stationary_experiment(runs=500, steps=1000):
    # 对比：样本平均 vs 固定步长 (alpha=0.1)
    # 两者都使用 epsilon=0.1 保证足够的探索
    labels = ['Sample-Average (1/n)', 'Constant Step-size (α=0.1)']
    all_rewards = np.zeros((2, steps))
    for r in range(runs):
        # 每次 run 初始化两个 agent 处理同一个环境
        env = NonStationaryBandit()
        agents = [Agent(alpha=None), Agent(alpha=0.1)]
        for s in range(steps):
            #env.step() # 环境发生变化！
            for i, agent in enumerate(agents):
                action = agent.select_action(method='gradient')
                reward = env.get_reward(action)
                agent.update_estimates(action, reward)
                all_rewards[i, s] += reward
                
    return all_rewards / runs

# 运行并绘图
avg_rewards = run_non_stationary_experiment()

plt.figure(figsize=(12, 6))
plt.plot(avg_rewards[0], label='Sample-Average', color='blue')
plt.plot(avg_rewards[1], label='Constant Step-size (α=0.1)', color='red')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Non-stationary Bandit: Sample-Average vs Constant Step-size')
plt.legend()
plt.show()
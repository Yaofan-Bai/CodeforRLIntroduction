import numpy as np
import matplotlib.pyplot as plt

# ==================== 环境类（不变） ====================
class NonStationaryBandit:
    def __init__(self, k=10):
        self.k = k
        self.true_values = np.random.randn(k)
        self.optimal_action = np.argmax(self.true_values)
    
    def step(self):
        self.true_values += np.random.normal(0, 0.01, self.k)
        self.optimal_action = np.argmax(self.true_values)
    
    def get_reward(self, action):
        return self.true_values[action] + np.random.randn()

# ==================== 基类（定义接口） ====================
class BanditAgent:
    """所有Agent的基类，定义统一接口"""
    def __init__(self, k=10, **kwargs):
        self.k = k
    
    def select_action(self):
        raise NotImplementedError
    
    def update(self, action, reward):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

# ==================== ε-greedy Agent ====================
class EpsilonGreedyAgent(BanditAgent):
    def __init__(self, k=10, epsilon=0.1, alpha=0.1, initial_q=0.0):
        super().__init__(k)
        self.epsilon = epsilon
        self.alpha = alpha  # 固定步长（适合非平稳）
        self.q_estimates = np.full(k, initial_q)
    
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q_estimates)
    
    def update(self, action, reward):
        self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])
    
    def reset(self):
        self.q_estimates.fill(0.0)

# ==================== UCB Agent ====================
class UCBAgent(BanditAgent):
    def __init__(self, k=10, c=2.0, alpha=0.1, initial_q=0.0):
        super().__init__(k)
        self.c = c  # 探索系数
        self.alpha = alpha
        self.q_estimates = np.full(k, initial_q)
        self.action_counts = np.zeros(k)
        self.total_steps = 0
    
    def select_action(self):
        self.total_steps += 1
        
        # 确保每个动作至少被选择一次
        if np.any(self.action_counts == 0):
            return np.random.choice(np.where(self.action_counts == 0)[0])
        
        # UCB公式
        ucb_values = self.q_estimates + self.c * np.sqrt(
            np.log(self.total_steps) / self.action_counts
        )
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])
    
    def reset(self):
        self.q_estimates.fill(0.0)
        self.action_counts.fill(0)
        self.total_steps = 0

# ==================== 梯度上升 Agent ====================
class GradientBanditAgent(BanditAgent):
    def __init__(self, k=10, alpha=0.1, baseline_type='reward_avg'):
        """
        baseline_type: 
            'zero' - 基线=0
            'q_mean' - 基线=Q估计均值
            'reward_avg' - 历史奖励平均（默认）
            'ema' - 指数移动平均
        """
        super().__init__(k)
        self.alpha = alpha
        self.baseline_type = baseline_type
        
        # 偏好和策略
        self.preferences = np.zeros(k)
        self.action_probs = np.ones(k) / k  # 初始均匀分布
        
        # 基线相关
        if baseline_type == 'q_mean':
            self.q_estimates = np.zeros(k)
            self.q_counts = np.zeros(k)
        elif baseline_type == 'reward_avg':
            self.baseline = 0.0
            self.step_count = 0
        elif baseline_type == 'ema':
            self.baseline = 0.0
            self.beta = 0.1  # EMA衰减率
        
    def select_action(self):
        # 计算softmax概率（数值稳定）
        exp_pref = np.exp(self.preferences - np.max(self.preferences))
        self.action_probs = exp_pref / np.sum(exp_pref)
        return np.random.choice(self.k, p=self.action_probs)
    
    def _calculate_baseline(self, action, reward):
        """根据类型计算基线"""
        if self.baseline_type == 'zero':
            return 0.0
        
        elif self.baseline_type == 'q_mean':
            # 更新Q估计
            self.q_counts[action] += 1
            step_size = 1.0 / self.q_counts[action] if self.q_counts[action] > 0 else 1.0
            self.q_estimates[action] += step_size * (reward - self.q_estimates[action])
            return np.mean(self.q_estimates)
        
        elif self.baseline_type == 'reward_avg':
            # 样本平均
            self.step_count += 1
            self.baseline += (reward - self.baseline) / self.step_count
            return self.baseline
        
        elif self.baseline_type == 'ema':
            # 指数移动平均
            self.baseline = self.beta * reward + (1 - self.baseline) * self.baseline
            return self.baseline
        
        else:
            return 0.0
    
    def update(self, action, reward):
        # 1. 计算基线
        baseline = self._calculate_baseline(action, reward)
        
        # 2. 更新偏好（梯度上升）
        for a in range(self.k):
            if a == action:
                self.preferences[a] += self.alpha * (reward - baseline) * (1 - self.action_probs[a])
            else:
                self.preferences[a] -= self.alpha * (reward - baseline) * self.action_probs[a]
    
    def reset(self):
        self.preferences.fill(0.0)
        self.action_probs = np.ones(self.k) / self.k
        if self.baseline_type == 'q_mean':
            self.q_estimates.fill(0.0)
            self.q_counts.fill(0)
        elif self.baseline_type == 'reward_avg':
            self.baseline = 0.0
            self.step_count = 0
        elif self.baseline_type == 'ema':
            self.baseline = 0.0

# ==================== 实验运行函数 ====================
def run_experiment(methods, runs=200, steps=1000):
    """
    运行对比实验
    methods: 要对比的方法列表，如 ['epsilon_greedy', 'ucb', 'gradient']
    """
    results = {
        'rewards': np.zeros((len(methods), steps)),
        'optimal_actions': np.zeros((len(methods), steps))
    }
    
    for run in range(runs):
        env = NonStationaryBandit()
        
        # 根据方法名创建Agent实例
        agents = []
        for method in methods:
            if method == 'epsilon_greedy':
                agents.append(EpsilonGreedyAgent(epsilon=0.1, alpha=0.1))
            elif method == 'ucb':
                agents.append(UCBAgent(c=2.0, alpha=0.1))
            elif method == 'gradient':
                agents.append(GradientBanditAgent(alpha=0.1, baseline_type='reward_avg'))
            elif method == 'gradient_q_mean':
                agents.append(GradientBanditAgent(alpha=0.1, baseline_type='q_mean'))
        
        for step in range(steps):
            #env.step()  # 环境变化
            
            for i, agent in enumerate(agents):
                action = agent.select_action()
                reward = env.get_reward(action)
                agent.update(action, reward)
                
                results['rewards'][i, step] += reward
                if action == env.optimal_action:
                    results['optimal_actions'][i, step] += 1
    
    # 平均
    results['rewards'] /= runs
    results['optimal_actions'] = results['optimal_actions'] / runs * 100  # 百分比
    
    return results

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 运行实验
    methods = ['epsilon_greedy', 'ucb', 'gradient', 'gradient_q_mean']
    results = run_experiment(methods, runs=2000, steps=1000)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['blue', 'green', 'red', 'orange']
    labels = ['ε-Greedy (α=0.1)', 'UCB (c=2.0)', 'Gradient (reward avg)', 'Gradient (Q mean)']
    
    # 平均奖励图
    ax = axes[0]
    for i, method in enumerate(methods):
        ax.plot(results['rewards'][i], label=labels[i], color=colors[i], alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('Non-stationary Bandit: Average Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 最优动作百分比图
    ax = axes[1]
    for i, method in enumerate(methods):
        ax.plot(results['optimal_actions'][i], label=labels[i], color=colors[i], alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Non-stationary Bandit: % Optimal Action')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印最终性能
    print("Final Performance (last 100 steps average):")
    for i, label in enumerate(labels):
        avg_reward = np.mean(results['rewards'][i, -100:])
        avg_optimal = np.mean(results['optimal_actions'][i, -100:])
        print(f"{label:20s} - Reward: {avg_reward:.4f}, Optimal %: {avg_optimal:.2f}%")
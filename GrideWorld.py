import numpy as np
from collections import defaultdict

class GridWorldEnvironment:
    def __init__(self, grid_size=(5,5), A=(0,1), B=(0,3), 
                 Aprime=(4,1), Bprime=(2,3), AReward=10, BReward=5):
        self.grid_size = grid_size
        self.A = A
        self.B = B
        self.Aprime = Aprime
        self.Bprime = Bprime
        self.AReward = AReward
        self.BReward = BReward
        
        self.StateValue = np.zeros(grid_size)
        self.optimal_policy = defaultdict(lambda: {a: 0.25 for a in ['up', 'down', 'left', 'right']})
        
        self.actions = ['up', 'down', 'left', 'right']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
    def state_transition(self, state, action):
        i, j = state
        
        # 特殊状态 A 和 B
        if state == self.A:
            return self.Aprime, self.AReward
        if state == self.B:
            return self.Bprime, self.BReward
        
        # 普通状态
        if action == 'up':
            next_state = (max(i-1, 0), j)
        elif action == 'down':
            next_state = (min(i+1, self.grid_size[0]-1), j)
        elif action == 'left':
            next_state = (i, max(j-1, 0))
        elif action == 'right':
            next_state = (i, min(j+1, self.grid_size[1]-1))
        else:
            next_state = state
            
        # 计算奖励
        if next_state == state and not (state == self.A or state == self.B):
            reward = -1
        else:
            reward = 0
            
        return next_state, reward
    
    def policy_evaluation_for_random_policy(self, gamma=0.9, theta=1e-6):
        """评估随机策略（每个动作概率相等）"""
        print("开始随机策略评估...")
        iteration = 0
        
        while True:
            delta = 0
            new_values = np.zeros(self.grid_size)
            
            # 预计算所有状态转移
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    state = (i, j)
                    v = 0
                    
                    # 对每个动作求和
                    for action in self.actions:
                        next_state, reward = self.state_transition(state, action)
                        ni, nj = next_state
                        v += 0.25 * (reward + gamma * self.StateValue[ni, nj])
                    
                    new_values[i, j] = v
                    delta = max(delta, abs(v - self.StateValue[i, j]))
            
            self.StateValue = new_values.copy()
            iteration += 1
            
            if delta < theta:
                print(f"随机策略评估在 {iteration} 次迭代后收敛")
                break
        
        return self.StateValue
    
    def value_iteration(self, gamma=0.9, theta=1e-6, max_iter=1000):
        """价值迭代算法：找到最优状态价值函数 V*(s)"""
        print("开始价值迭代寻找最优解...")
        iteration = 0
        
        while iteration < max_iter:
            delta = 0
            new_values = np.zeros(self.grid_size)
            
            # 一次性计算所有状态的新价值
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    state = (i, j)
                    
                    # 计算每个动作的Q值
                    max_q = -float('inf')
                    for action in self.actions:
                        next_state, reward = self.state_transition(state, action)
                        ni, nj = next_state
                        q = reward + gamma * self.StateValue[ni, nj]
                        if q > max_q:
                            max_q = q
                    
                    # 贝尔曼最优方程：V*(s) = max_a Q(s,a)
                    new_values[i, j] = max_q
                    delta = max(delta, abs(max_q - self.StateValue[i, j]))
            
            self.StateValue = new_values.copy()
            iteration += 1
            
            if delta < theta:
                print(f"价值迭代在 {iteration} 次迭代后收敛")
                break
        
        if iteration >= max_iter:
            print(f"达到最大迭代次数 {max_iter}")
        
        # 从最优状态价值中提取最优策略
        self.extract_optimal_policy_from_values(gamma)
        
        return self.StateValue
    
    def extract_optimal_policy_from_values(self, gamma=0.9):
        """从最优状态价值函数中提取最优策略（优化版本）"""
        self.optimal_policy = defaultdict(dict)
        
        # 预先计算所有状态的Q值
        Q_matrix = {}
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state = (i, j)
                state_q_values = []
                
                for action in self.actions:
                    next_state, reward = self.state_transition(state, action)
                    ni, nj = next_state
                    q = reward + gamma * self.StateValue[ni, nj]
                    state_q_values.append((q, action))
                
                Q_matrix[state] = state_q_values
        
        # 一次性更新所有状态的策略
        for state in Q_matrix:
            q_list = Q_matrix[state]
            
            # 找出最大Q值的动作
            max_q = max(v for v, _ in q_list)
            best_actions = [a for v, a in q_list if v == max_q]
            
            # 分配概率（均匀分布给所有最优动作）
            prob = 1.0 / len(best_actions)
            for action in self.actions:
                self.optimal_policy[state][action] = prob if action in best_actions else 0.0
    
    def compute_optimal_action_values(self, gamma=0.9):
        """计算最优动作价值函数 Q*(s,a)"""
        Q_star = {}
        
        # 一次性计算所有状态-动作对的Q值
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state = (i, j)
                state_q_values = np.zeros(len(self.actions))
                
                for idx, action in enumerate(self.actions):
                    next_state, reward = self.state_transition(state, action)
                    ni, nj = next_state
                    state_q_values[idx] = reward + gamma * self.StateValue[ni, nj]
                
                Q_star[state] = state_q_values
        
        return Q_star
    
    def print_state_values(self, title="状态价值函数"):
        """打印状态价值函数"""
        print(f"\n=== {title} ===")
        print("(0,0)在左上角，(4,4)在右下角")
        
        # 从第一行开始打印（左下角为原点）
        for i in range(self.grid_size[0]):
            row_str = ""
            for j in range(self.grid_size[1]):
                row_str += f"{self.StateValue[i, j]:6.2f}"
            print(row_str)
    
    def print_optimal_policy(self):
        """打印最优策略"""
        print("\n=== 最优策略 π*(s) ===")
        print("箭头表示动作方向")
        
        # 动作符号映射
        action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
        
        # 从第一行开始打印
        for i in range(self.grid_size[0]):
            row_str = ""
            for j in range(self.grid_size[1]):
                state = (i, j)
                policy = self.optimal_policy[state]
                
                # 找出概率大于0的动作
                active_actions = [action_symbols[a] for a, p in policy.items() if p > 0]
                
                if not active_actions:
                    row_str += "  ·  "
                elif len(active_actions) == 1:
                    row_str += f"  {active_actions[0]}  "
                elif len(active_actions) == 2:
                    row_str += f" {active_actions[0]}{active_actions[1]} "
                elif len(active_actions) == 3:
                    row_str += f"{active_actions[0]}{active_actions[1]}{active_actions[2]}"
                elif len(active_actions) == 4:
                    row_str += "↻"  # 所有方向都最优
                else:
                    row_str += "  ?  "
                
                row_str += " "
            
            print(row_str)
    
    def run_analysis(self, gamma=0.9):
        """运行完整分析"""
        print("=" * 50)
        print("GridWorld MDP 分析")
        print("=" * 50)
        
        # 1. 随机策略评估
        print("\n1. 随机策略评估 (对应图3.2右图)")
        env_random = GridWorldEnvironment()
        env_random.policy_evaluation_for_random_policy(gamma=gamma)
        env_random.print_state_values("随机策略状态价值")
        
        # 2. 价值迭代
        print("\n" + "-" * 50)
        print("\n2. 价值迭代寻找最优解")
        env_optimal = GridWorldEnvironment()
        env_optimal.value_iteration(gamma=gamma)
        
        # 打印结果
        env_optimal.print_state_values("最优状态价值 V*(s)")
        env_optimal.print_optimal_policy()
        
        # 3. 最优动作价值
        print("\n" + "-" * 50)
        print("\n3. 最优动作价值 Q*(s,a)")
        Q_star = env_optimal.compute_optimal_action_values(gamma)
        
        # 打印示例
        sample_states = [(0, 0), (0, 2), (2, 2), (4, 0), (4, 4)]
        for state in sample_states:
            i, j = state
            print(f"\n状态 ({i},{j}):")
            for idx, action in enumerate(env_optimal.actions):
                print(f"  {action:6}: {Q_star[state][idx]:7.3f}")
        
        return env_random, env_optimal


if __name__ == "__main__":
    # 运行分析
    env_random, env_optimal = GridWorldEnvironment().run_analysis(gamma=0.9)
    
    print("\n" + "=" * 50)
    print("分析完成！")
    print("=" * 50)
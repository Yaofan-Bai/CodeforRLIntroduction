import numpy as np
import math
import matplotlib.pyplot as plt
class JacksCarRentalEnvironment:
    def __init__(self, MaxCarsA=20, MaxCarsB=20, MaxMove=5, RentReward=10, MoveCost=2,
                  LambdaRentA=3, LambdaRentB=4, LambdaReturnA=3, LambdaReturnB=2, MAX_DEMAND=20):
        self.MaxCarsA = MaxCarsA
        self.MaxCarsB = MaxCarsB
        self.MaxMove = MaxMove
        self.RentReward = RentReward
        self.MoveCost = MoveCost
        self.LambdaRentA = LambdaRentA
        self.LambdaRentB = LambdaRentB
        self.LambdaReturnA = LambdaReturnA
        self.LambdaReturnB = LambdaReturnB
        self.StateValue = np.zeros((MaxCarsA + 1, MaxCarsB + 1))
        self.Policy = np.zeros((MaxCarsA + 1, MaxCarsB + 1), dtype=int)
        self.Actions = np.arange(-MaxMove, MaxMove + 1)
        self.MAX_DEMAND = MAX_DEMAND
    def Posson(self, n, lam):
        """计算 Possion 分布概率"""
        return (lam ** n) * np.exp(-lam) / math.factorial(n) #需要求阶乘，使用math库的factorial函数，后续使用查表优化性能
    def PossonTail(self, n, lam):
        """计算 Possion 分布的尾概率 P(X >= n)"""
        return 1 - sum(self.Posson(k, lam) for k in range(n))
    def TransitionProb(self, State, action):
        """计算从当前状态执行动作到达下一个状态的转移概率"""
        probs = {}  # key: (s_prime, reward), value: probability
        carsA, carsB = State
        #合法性检查
        carsAafterMove = carsA - action
        carsBafterMove = carsB + action
        if carsAafterMove < 0 or carsAafterMove > self.MaxCarsA:
            return probs
        if carsBafterMove < 0 or carsBafterMove > self.MaxCarsB:
            return probs
        #挪车惩罚计算
        move_cost = -abs(action) * self.MoveCost
        # 枚举rentA, rentB,所有可能组合，计算对应的概率和奖励
        for rentA in range(0, carsAafterMove + 1):
            for rentB in range(0, carsBafterMove + 1):
                if rentB < carsBafterMove:
                    prob_rentB = self.Posson(rentB, self.LambdaRentB)
                else:
                    prob_rentB = self.PossonTail(carsBafterMove, self.LambdaRentB)
                if rentA < carsAafterMove:
                    prob_rentA = self.Posson(rentA, self.LambdaRentA)
                else:
                    prob_rentA = self.PossonTail(carsAafterMove, self.LambdaRentA)
                rent_reward = (rentA + rentB) * self.RentReward
                # rent 之后剩余库存
                carsAafterRent = carsAafterMove - rentA
                carsBafterRent = carsBafterMove - rentB
                maxReturnA = self.MaxCarsA - carsAafterRent
                maxReturnB = self.MaxCarsB - carsBafterRent
                    #枚举returnA, returnB，所有可能组合，计算对应的概率和奖励
                for returnA in range(0, maxReturnA + 1):
                    for returnB in range(0, maxReturnB + 1):
                        if returnB < maxReturnB:
                            prob_returnB = self.Posson(returnB, self.LambdaReturnB)
                        else:
                            prob_returnB = self.PossonTail(maxReturnB, self.LambdaReturnB)
                        if returnA < maxReturnA:
                            prob_returnA = self.Posson(returnA, self.LambdaReturnA)
                        else:
                            prob_returnA = self.PossonTail(maxReturnA, self.LambdaReturnA) 
                        #计算下一个状态和奖励
                        nextCarsA = carsAafterRent + returnA
                        nextCarsB = carsBafterRent + returnB
                        next_state = (nextCarsA, nextCarsB)
                        #reward = 租车奖励 - 挪车惩罚
                        reward = rent_reward + move_cost
                        #计算转移概率
                        prob = prob_rentA * prob_rentB * prob_returnA * prob_returnB
                        if (next_state, reward) not in probs:
                            probs[(next_state, reward)] = 0.0
                        probs[(next_state, reward)] += prob
        return probs
    def ExpectedReturn(self, State, action,gamma=0.9):
        """计算给定状态和动作的期望回报"""
        expected_value = 0
        probs = self.TransitionProb(State, action)
        if not probs:
            return float('-inf')
        for (next_state, reward), prob in probs.items():
            NextCarsA, NextCarsB = next_state
            expected_value += prob * (reward + gamma * self.StateValue[NextCarsA, NextCarsB]) 
        return expected_value
        
    def PolicyEvaluation(self, gamma=0.9, theta=1e-6):
        """评估当前策略"""
        print("开始策略评估...")
        iteration = 0
        while True:
            delta = 0
            new_values = np.zeros_like(self.StateValue)
            for carsA in range(self.MaxCarsA + 1):
                for carsB in range(self.MaxCarsB + 1):
                    v = 0
                    action = self.Policy[carsA, carsB]
                    v = self.ExpectedReturn((carsA, carsB), action, gamma)
                    new_values[carsA, carsB] = v
                    delta = max(delta, abs(v-self.StateValue[carsA, carsB]))
            self.StateValue = new_values.copy()
            if delta < theta:
                print(f"策略评估在第 {iteration} 次迭代后收敛")
                self.PrintValues()
                return
            iteration += 1
            print(f"进行第 {iteration} 次策略评估，残差为{delta}")
            #self.PrintValues()
    def PolicyImprovement(self, gamma=0.9):
        """改进当前策略"""
        print("开始策略改进...")
        policy_stable = True
        for carsA in range(self.MaxCarsA + 1):
            for carsB in range(self.MaxCarsB + 1):
                old_action = self.Policy[carsA, carsB]
                action_returns = []
                for action in self.Actions:
                    if (action <= carsA) and (-action <= carsB):
                        action_returns.append(self.ExpectedReturn((carsA, carsB), action, gamma))
                    else:
                        action_returns.append(float('-inf'))  # 不合法动作
                max_return = max(action_returns)
                best_actions = [a for a, r in zip(self.Actions, action_returns) if r == max_return]
                new_action = np.random.choice(best_actions)
                self.Policy[carsA, carsB] = new_action
                #print(f"状态 ({carsA}, {carsB}) 的最佳动作: {new_action} (旧动作: {old_action})")
                if old_action != new_action:
                    policy_stable = False
        if policy_stable:
            print("策略已稳定，不需要进一步改进")
        else:
            print("策略已更新，需要继续评估")
        return policy_stable
    def PrintPolicy(self):
        plt.figure(figsize=(10,8))
        im = plt.imshow(self.Policy.T, origin='lower', cmap='coolwarm', interpolation='nearest')
        plt.colorbar(im)
        plt.xlabel('Number of cars at A')
        plt.ylabel('Number of cars at B')
        plt.title('Policy Heatmap for Jack\'s Car Rental')
        plt.show()
    def PrintValues(self):
        """打印当前状态价值"""
        print("当前状态价值:")
        for carsA in range(self.MaxCarsA + 1):
            row_str = ""
            for carsB in range(self.MaxCarsB + 1):
                row_str += f"{self.StateValue[carsA, carsB]:.2f} "
            print(row_str)
    def RunAnalysis(self, gamma=0.9, theta=1e-6):
        """执行策略迭代算法"""
        print("\n" + "-" * 50)
        print("\n2. 价值迭代寻找最优解")
        while True:
            self.PolicyEvaluation(gamma=gamma, theta=theta)
            stable = self.PolicyImprovement(gamma=gamma)
            if stable:
                break

if __name__ == "__main__":
    env = JacksCarRentalEnvironment()
    env.RunAnalysis()
    env.PrintPolicy()               
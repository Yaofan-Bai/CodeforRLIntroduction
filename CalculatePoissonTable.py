import numpy as np
import math
import pandas as pd
class PoissonTable:
    def __init__(self, max_n=20, Lambda=[3, 4, 2]):
        self.max_n = max_n
        self.Lambda = Lambda
        self.table = np.zeros((max_n + 1, len(self.Lambda)))
        self._calculate_Poisson_table()
        self.tail_table = np.zeros((max_n + 1, len(self.Lambda)))
        self._calculate_Poisson_tail_table()
        self.save_poisson_table()
        self.save_poisson_tail_table()
    def _calculate_Poisson_table(self):
        """预计算 Poisson 分布概率表"""
        for lam_idx, lam in enumerate(self.Lambda):
            for n in range(self.max_n + 1):
                self.table[n, lam_idx] = (lam ** n) * np.exp(-lam) / math.factorial(n)
    def _calculate_Poisson_tail_table(self):
        """预计算 Poisson 分布尾概率表"""
        for lam_idx, lam in enumerate(self.Lambda):
            for n in range(self.max_n + 1):
                self.tail_table[n, lam_idx] = 1 - np.sum(self.table[:n+1, lam_idx])
    def get_probability(self, n, lam):
        """获取 Possion 分布概率 P(X=n)"""
        if n > self.max_n or lam not in self.Lambda:
            raise ValueError("n 或 lambda 超出预计算范围")
        lam_idx = self.Lambda.index(lam)
        return self.table[n, lam_idx]
    def get_tail_probability(self, n, lam):
        """获取 Possion 分布尾概率 P(X>=n)"""
        if n > self.max_n or lam not in self.Lambda:
            raise ValueError("n 或 lambda 超出预计算范围")
        lam_idx = self.Lambda.index(lam)
        return 1 - np.sum(self.table[:n+1, lam_idx])
    
    def save_poisson_table(self, filename='poisson_table.csv'):
        """保存泊松分布概率表为CSV文件"""
        # 创建列名
        columns = [f'lambda={lam}' for lam in self.Lambda]
        df = pd.DataFrame(self.table, columns=columns)
        df.index.name = 'n'
        df.to_csv(filename)
        print(f"泊松分布表已保存到 {filename}")
    
    def save_poisson_tail_table(self, filename='poisson_tail_table.csv'):
        """保存泊松分布尾概率表为CSV文件"""
        # 创建列名
        columns = [f'lambda={lam}' for lam in self.Lambda]
        df = pd.DataFrame(self.tail_table, columns=columns)
        df.index.name = 'n'
        df.to_csv(filename)
        print(f"泊松分布尾概率表已保存到 {filename}")
def main():
    # 创建 Poisson 表实例
    poisson_table = PoissonTable(max_n=20, Lambda=[3, 4, 2])
    # 示例：获取 P(X=5) 和 P(X>=5) 对于 lambda=3 的概率
    print(f"P(X=5|λ=3) = {poisson_table.get_probability(5, 3):.6f}")
    print(f"P(X>=5|λ=3) = {poisson_table.get_tail_probability(5, 3):.6f}")

if __name__ == '__main__':
    main()

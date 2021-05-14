# encoding: utf-8
import data_create
import numpy as np
from mopso import *
from time import *

def main():
    begin_time = time()
    # 参数初始化
    w1 = 0.4  # 惯性因子极小值
    w2 = 0.9  # 惯性因子极大值
    c1 = 2  # 局部速度因子
    c2 = 2  # 全局速度因子
    theta0 = 0.3
    particals = 100  # 粒子群的数量
    cycle_ = 300  # 迭代次数
    mesh_div = 10  # 网格等分数量
    thresh = 300  # 外部存档阀值

    data = data_create.data_model()
    mopso_ = Mopso(data, particals, w1, w2, c1, c2, thresh, theta0, mesh_div)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(cycle_)  # 经过cycle_轮迭代后，pareto边界粒子
    np.savetxt("./img_txt/pareto_in.txt", pareto_in)  # 保存pareto边界粒子的坐标
    np.savetxt("./img_txt/pareto_fitness.txt", pareto_fitness)  # 打印pareto边界粒子的适应值
    print("\n", "pareto边界的坐标保存于：/img_txt/pareto_in.txt")
    print("pareto边界的适应值保存于：/img_txt/pareto_fitness.txt")
    print("\n", "迭代结束,over")

    end_time = time()
    run_time = end_time - begin_time
    print(run_time)

if __name__ == "__main__":
    main()
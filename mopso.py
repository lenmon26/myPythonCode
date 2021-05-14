# encoding: utf-8
import numpy as np
from fitness_funs import *
import init
import update
import plot


class Mopso:
    def __init__(self, data, particals, w1, w2, c1, c2, thresh, theta0, mesh_div=10):
        self.w1, self.w2, self.c1, self.c2, self.theta0 = w1, w2, c1, c2, theta0
        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.data = data
        self.nodes = self.data["nodes"]
        self.vehicles = self.data["vehicles"]
        self.in_min = self.data['extreme_position'][0]
        self.in_max = self.data['extreme_position'][1]
        self.v_min = self.data['extreme_velocity'][0]
        self.v_max = self.data['extreme_velocity'][1]
        self.plot_ = plot.Plot_pareto()
    
    def evaluation_fitness(self):
        # 计算适应值
        fitness_curr = []
        # 是否需要放在前面初始化函数里面
        for i in range(self.in_.shape[0]):
            fit = Fitness(self.data, self.in_[i])
            fitness_curr.append(fit.fitness_())   # 每次调用只计算第i个粒子的适用度，in_[i]是一维数组
        self.fitness_ = np.array(fitness_curr)    # 适应值

    def initialize(self):
        # 初始化粒子坐标
        self.in_ = init.init_designparams(self.particals, self.nodes)   # 整数（表示各节点的次序）
        # 初始化粒子速度
        self.v = init.init_v(self.particals, self.nodes, self.v_max, self.v_min)   # 浮点数
        # 计算适应值
        self.evaluation_fitness()
        # 初始化个体最优
        self.in_p, self.fitness_p = init.init_pbest(self.in_, self.fitness_)
        # 初始化外部存档
        self.archive_in, self.archive_fitness = init.init_archive(self.in_, self.fitness_)
        # 初始化全局最优
        self.in_g, self.fitness_g = init.init_gbest(self.archive_in,self.archive_fitness, self.mesh_div, self.particals)

    def update_(self):
        # 更新粒子速度、粒子坐标、适应值、个体最优、外部存档、全局最优
        self.v = update.update_v(self.v, self.in_, self.in_p, self.in_g, self.fitness_, self.fitness_g, self.k, self.K)
        self.in_ = update.update_in(self.in_, self.v, self.nodes)
        self.evaluation_fitness()
        self.in_p, self.fitness_p = update.update_pbest(self.in_, self.fitness_, self.in_p, self.fitness_p)
        self.archive_in, self.archive_fitness = update.update_archive(self.in_, self.fitness_, self.archive_in, self.archive_fitness, self.thresh, self.mesh_div, self.particals)
        self.in_g, self.fitness_g = update.update_gbest(self.in_, self.fitness_, self.archive_in, self.archive_fitness, self.mesh_div, self.particals)

    def done(self, cycle_):
        self.initialize()
        self.plot_.show1(self.fitness_, self.archive_fitness, -1)
        self.K = cycle_
        archive_fit1, archive_fit2, archive_fit3 =\
        list(self.archive_fitness[:, 0]), list(self.archive_fitness[:, 1]), list(self.archive_fitness[:, 2])
        for i in range(cycle_):
            self.k = i
            self.update_()
            if (i+1) % 5 == 0:
                self.plot_.show1(self.fitness_, self.archive_fitness, i)
            if (i+1) % (cycle_/10) == 0:
                archive_fit1.append(self.archive_fitness[:, 0])
                archive_fit2.append(self.archive_fitness[:, 1])
                archive_fit3.append(self.archive_fitness[:, 2])
        self.plot_.show2(archive_fit1, archive_fit2, archive_fit3, cycle_)
        return self.archive_in, self.archive_fitness

# encoding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fitness_funs import *

class Plot_pareto:
    def __init__(self):
        if os.path.exists('./img_txt') == False:
            os.makedirs('./img_txt')
            print('创建文件夹img_txt:保存粒子群每一次迭代的图片')

    def show1(self,fit,arc_fit, i):
        # 展示pareto边界的形成过程
        clear_index = []
        for j in range(len(fit)):
            if (100000 in fit[j]):
                clear_index.append(j)
        fit = np.delete(fit, clear_index, 0)
        fig = plt.figure()
        f_max = np.max(np.array(fit), axis=0)  # 求网格的取值范围
        f_min = np.min(np.array(fit), axis=0)
        fd = f_max - f_min
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('fitness_time')
        ax.set_ylabel('fitness_cost')
        ax.set_zlabel('fitness_carb')
        ax.scatter((fit[:, 0]-f_min[0])/fd[0], (fit[:, 1]-f_min[1])/fd[1], (fit[:, 2]-f_min[2])/fd[2], s = 20, c = 'blue', marker = ".")
        ax.scatter((arc_fit[:, 0]-f_min[0])/fd[0], (arc_fit[:, 1]-f_min[1])/fd[1], (arc_fit[:, 2]-f_min[2])/fd[2], s = 50, c = 'red', marker = ".")
        # plt.show()
        plt.savefig('./img_txt/第' + str(i + 1) + '次迭代.png')
        print('第' + str(i + 1) + '次迭代的图片保存于 img_txt 文件夹')
        plt.close()

    def show2(self, archive_fit1,archive_fit2,archive_fit3, K):
        # 共3个子图，第1、2、3子图绘制输入迭代次数与适应值1、2、3之间的关系
        fig = plt.figure(13, figsize=(17, 5))
        ax1 = fig.add_subplot(131)
        ax1.set_xlabel('step')
        ax1.set_ylabel('fitness_time')
        for i in range(11):
            num = archive_fit1[i].shape
            step = np.ones(num) * i * (K / 10)
            ax1.scatter(step, archive_fit1[i], s = 20)
        ax2 = fig.add_subplot(132)
        ax2.set_xlabel('step')
        ax2.set_ylabel('fitness_cost')
        for i in range(11):
            num = archive_fit2[i].shape
            step = np.ones(num) * i * (K / 10)
            ax2.scatter(step, archive_fit2[i], s = 20)
        ax3 = fig.add_subplot(133)
        ax3.set_xlabel('step')
        ax3.set_ylabel('fitness_carb')
        for i in range(11):
            num = archive_fit3[i].shape
            step = np.ones(num) * i * (K / 10)
            ax3.scatter(step, archive_fit3[i], s = 20)
        # plt.show()
        plt.savefig('./img_txt/适应度函数1-3.png')
        print('适应度函数1-3图片保存于 img_txt 文件夹')
        plt.close('all')  # 避免内存泄漏
        plt.close()

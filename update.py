#encoding: utf-8
import numpy as np
import random
import pareto
import archiving

w1,w2,c1,c2,theta0,m,n = 0.4,0.9,2.0,2.0,0.3,0.15,0.3

def update_v(v_,in_,in_pbest,in_gbest,f_x,f_gbest,k,K):
    #更新速度值
    f_g = np.min(np.array(f_gbest), axis=0)
    v_temp = []
    for i in range(v_.shape[0]):
        w = w1 + (w2-w1)* k / K
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        h1 = (f_g[0]/f_x[i][0] + f_g[1]/f_x[i][1] + f_g[2]/f_x[i][2]) / 3
        h2 = (K-k)/K
        pb = m * (h1+h2) + n
        phi = random.uniform(0,1)
        if (phi >= pb):
            theta1, theta2 = 1,theta0
        else:
            theta1, theta2 = theta0,1
        v_temp1 = w * v_[i] + c1 * r1 * theta1* (in_pbest[i]-in_[i]) + c2 * r2 * theta2 * (in_gbest[i]-in_[i])
        v_temp.append(v_temp1)
    v_temp = np.array(v_temp)
    return v_temp
def update_in(in_,v_,nodes):
    #更新位置参数
    in_temp = in_ + v_
    #重新排序,把in_修正为1~nodes区间
    for i in range(in_temp.shape[0]):
        temp1 = in_temp[i]
        temp2 = temp1.copy()
        temp2[0] = -1000
        temp2.sort()
        temp2[0] = 0
        for j in range(1, nodes):
            k = np.where(temp1 == temp2[j])
            temp1[k] = j
        in_temp[i] = temp1
    return in_temp
def compare_pbest(in_indiv,pbest_indiv):
    num_greater = 0
    num_less = 0
    for i in range(len(in_indiv)):
        if in_indiv[i] > pbest_indiv[i]:
            num_greater = num_greater +1
        if in_indiv[i] < pbest_indiv[i]:
            num_less = num_less +1
    #如果历史pbest支配当前粒子，则不更新
    if (num_greater>0 and num_less==0):
        return False
    #如果当前粒子支配历史pbest，则更新历史pbest
    elif (num_greater==0 and num_less>0):
        return True
    #如果互不支配，则按照概率决定是否更新
    else:
        random_ = random.uniform(0.0,1.0)
        if random_ > 0.5:
            return True
        else:
            return False
def update_pbest(in_,fitness_,in_pbest,fitness_pbest):
    for i in range(fitness_pbest.shape[0]):
        #通过比较历史pbest和当前粒子适应值，决定是否需要更新pbest的值。
        if compare_pbest(fitness_[i],fitness_pbest[i]):
            fitness_pbest[i] = fitness_[i]
            in_pbest[i] = in_[i]
    return in_pbest,fitness_pbest
def update_archive(in_,fitness_,archive_in,archive_fitness,thresh,mesh_div,particals):
    #首先，计算当前粒子群的pareto边界，将边界粒子加入到存档archiving中
    pareto_1 = pareto.Pareto_(in_,fitness_)
    curr_in,curr_fit = pareto_1.pareto()
    #其次，在存档中根据支配关系进行第二轮筛选，将非边界粒子去除
    in_new = np.concatenate((archive_in,curr_in),axis=0)
    fitness_new = np.concatenate((archive_fitness,curr_fit),axis=0)
    pareto_2 = pareto.Pareto_(in_new,fitness_new)
    curr_archiving_in,curr_archiving_fit = pareto_2.pareto()
    #最后，判断存档数量是否超过了存档阀值。如果超过了阀值，则清除掉一部分（拥挤度高的粒子被清除的概率更大）
    if((curr_archiving_in).shape[0] > thresh):
        clear_ = archiving.clear_archiving(curr_archiving_in,curr_archiving_fit,mesh_div,particals)
        curr_archiving_in,curr_archiving_fit = clear_.clear_(thresh)
    return curr_archiving_in,curr_archiving_fit
def update_gbest(in_,fitness_,archiving_in,archiving_fit,mesh_div,particals):
    random_ = random.uniform(0.0,1.0)
    if random_ > 0.8:
        return in_,fitness_
    else:
        get_g = archiving.get_gbest(archiving_in,archiving_fit,mesh_div,particals)
        return get_g.get_gbest()
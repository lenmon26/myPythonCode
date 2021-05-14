# encoding: utf-8
import random
import numpy as np
from numpy import ndarray

import archiving
import pareto
import math
def init_designparams(particals,nodes):
    '''粒子位置初始化:（particals，nodes）共particals个粒子，每个粒子有L维'''
    in_=np.zeros((particals,nodes))
    for i in range(particals):
        in_[i] = range(nodes)
        random.shuffle(in_[i][1:])
    return in_
def init_v(particals,nodes,v_max,v_min):
    '''粒子速度初始化:（particals，nodes）共particals个粒子，每个粒子有L维'''
    v_in = np.zeros((particals,nodes))
    for i in range(particals):
        for j in range(1,nodes):
            v_in[i,j] = random.uniform(0,1)*(v_max-v_min)+v_min   
    return v_in
def init_pbest(in_,fitness_):
    return in_,fitness_
def init_archive(in_,fitness_):
    pareto_c = pareto.Pareto_(in_,fitness_)
    curr_archiving_in,curr_archiving_fit = pareto_c.pareto()
    return curr_archiving_in,curr_archiving_fit
def init_gbest(curr_archiving_in,curr_archiving_fit,mesh_div,particals):
    get_g = archiving.get_gbest(curr_archiving_in,curr_archiving_fit,mesh_div,particals)
    return get_g.get_gbest()
	
	####ljq test upload
	#add plus
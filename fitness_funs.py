# encoding: utf-8
import numpy as np
import math

class Fitness:
    def __init__(self, data, in_):
        self.in_ = in_                                       # in_是第i个粒子所表示的各节点的排序，是一维数组
        self.data = data
        self.locations = self.data["locations"]              # 节点位置
        self.nodes = self.data["nodes"]                      # 需求节点数（含配送中心）
        self.time_windows = self.data["time_windows"]        # 节点时间窗
        self.service_time = self.data["service_time"]        # 节点服务时间
        self.demands = self.data["demands"]                  # 节点需求量
        self.capacities = self.data["vehicle_capacities"]    # 车辆容量
        self.vehicles = self.data["vehicles"]                # 可调度车辆数
        self.unit_trans_cost = self.data['unit_trans_cost']  # 配送中心到需求点，车辆的单位运输成本，单位：元/公里
        self.vehicle_useCost = self.data['vehicle_useCost']  # 车辆的使用成本，单位：元
        self.vehicle_speed = self.data['vehicle_speed']      # 配送中心到需求点，车辆的单位行驶速度，单位：公里/小时
        self.unit_ET_cost = self.data['unit_ET_cost']        # 早到机会惩罚成本系数
        self.unit_LT_cost = self.data['unit_LT_cost']        # 不满足时间窗要求的晚到单位惩罚成本
        self.unit_loss_cost = self.data['unit_loss_cost']    # 单位货损成本系数
        self.in_min = self.data['extreme_position'][0]       # 1
        self.in_max = self.data['extreme_position'][1]       # L（nodes-1)
        self.v_min = self.data['extreme_velocity'][0]        # -(nodes-2)
        self.v_max = self.data['extreme_velocity'][1]        # (nodes-2)
        self.speed = self.data['vehicle_speed']              # 车辆平均速度
        self.distance = self.data["distance"]

    def x_y(self):
        '''计算当前粒子的x[k][i][j],y[k][j]'''
        x = np.zeros([self.vehicles+1,self.nodes,self.nodes])
        y = np.zeros([self.vehicles+1,self.nodes])
        load = np.zeros([self.vehicles+2])
        k = 1
        for m in range(1,self.nodes):
            if (load[k] == 0):
                i = 0
                j = self.in_[m]
            else:
                i = self.in_[m-1]
                j = self.in_[m]
            i = int(i)
            j = int(j)
            x[k][i][j] = 1
            y[k][j] = 1
            load[k] = load[k] + self.demands[j]  
            if (load[k] > self.capacities):# 车辆不要超重
                load[k] = load[k] - self.demands[j]                
                x[k][i][j] = x[k][i][j]-1
                y[k][j] = y[k][j]-1
                x[k][i][0] = 1
                i = 0
                j = self.in_[m]
                j = int(j)
                k = k + 1
                if (k > self.vehicles):
                    return x, y, load,0
                else:
                    x[k][i][j] = 1
                    y[k][0] = 1
                    y[k-1][0] = 1
                    y[k][j] = 1
                    load[k] = load[k] + self.demands[j]
            if (m == (self.nodes - 1)):
                i = self.in_[m]
                i = int(i)
                j = 0
                x[k][i][j] = 1
        return x, y, load,1
    
    def QLJ(self):
        '''J节点剩余载重'''
        QLL = 0
        QL = np.zeros([self.nodes])
        for k in range(1,self.vehicles+1):
            if self.load[k] == 0:
                break
            i = 0
            while 1:
                for j in range(1,self.nodes):
                    if (self.x[k][i][j] == 1):
                        if (i == 0):
                            QL[j] = self.load[k] - self.demands[j]
                            QLL = QL[j]
                            break
                        else:
                            QL[j] = QLL - self.demands[j]
                            QLL = QL[j]
                            break
                i = j
                if (QL[j] == 0):
                    break
        self.QL = QL
    
    def fitness_time(self):
        '''时间总费用'''
        time = np.zeros([ self.vehicles +1 ])
        start_time = np.zeros([ self.vehicles +1 ])
        arrive_time = np.zeros([ self.nodes ])
        wait_time = np.zeros([ self.nodes ])
        travel_time = np.zeros([ self.nodes ])
        punish_time = np.zeros([ self.nodes ])
        i = 0
        m = 1
        for k in range(1,self.vehicles + 1):
            if self.load[k] == 0:                    #所用车辆数小于可用车辆数
                break
            while 1:
                if ((i == 0) and (k == 1) ):
                    j = self.in_[m]                  # 序列中第一个节点特殊处理，作为第一段的目的地
                    j = int(j)
                if (self.x[k][i][j] == 0 ):          # i节点已是k辆车行驶路线最后一个节点
                    j = 0
                if (self.x[k][i][j] == 1 ):
                    travel_time[j] = self.x[k][i][j] * (self.distance[i][j] / self.speed) * 60           # 节点i，j行驶时间
                    arrive_time[j]  = time[k] + travel_time[j]
                    if( i == 0):                    # k辆车行驶路线第一个节点，出发时间特殊处理
                        start_time[k] = max((self.time_windows[j][0] - arrive_time[j]),0) 
                        arrive_time[j] = self.time_windows[j][0]
                    wait_time[j] = max((self.time_windows[j][0] - arrive_time[j]),0)                    # 客户等待时间
                    punish_time[j] = self.unit_ET_cost * (max((self.time_windows[j][0] - arrive_time[j]), 0)) + \
                                     self.unit_LT_cost * (max((arrive_time[j] - self.time_windows[j][1]), 0))
                    time[k] = arrive_time[j] + wait_time[j] + self.service_time[j]

                if (j == 0):      #车辆返回配送中心
                    if(m <= self.nodes-1):   #
                        i = 0
                        j = self.in_[m]
                        j = int(j)
                        break
                    else:
                        break
                else:
                    i = j
                    m += 1
                    if (m <= (self.nodes - 1)):
                        j = self.in_[m]
                        j = int(j)
                    else:
                        j= 0
        self.start_time = start_time
        self.travel_time = travel_time
        self.arrive_time = arrive_time
        self.wait_time = wait_time
        total_time = sum(time) + sum(punish_time)- sum(start_time)
        return total_time
                
    def fitness_cost(self):
        '''货损量'''
        C1 = 200
        C2 = 3
        P = 2000
        a1 = 0.002
        a2 = 0.003
        Ce1 = 15
        Ce2 = 20
        '''Z1.固定成本'''
        Z1 = 0
        j = 0
        for k in range(1,self.vehicles + 1):
            if self.load[k] == 0:
                break
            if (self.y[k][j] == 1):
                Z1 += self.y[k][j]
        Z1 *= C1
        '''Z2: 运输成本'''
        Z2 = 0 
        for k in range(1,self.vehicles + 1):
            if self.load[k]==0:
                break
            for i in range(self.nodes):
                for j in range(self.nodes):
                    if (self.x[k][i][j] == 1):
                        Z2 += self.distance[i][j] * self.x[k][i][j]
                        break
        Z2 *= C2
        '''Z3: 货损成本'''
        Z3 = 0 
        self.QLJ()
        for k in range(1,self.vehicles + 1):
            if self.load[k] == 0:
                break
            for j in range(1,self.nodes):
                if (self.y[k][j] == 1):
                    z31 = 1-  math.exp(-a1 * (self.arrive_time[j] + self.wait_time[j] - self.start_time[k])/60)
                    z32 = 1 - math.exp((-a2) * self.service_time[j]/60)
                    Z3 += self.y[k][j] * (self.demands[j] * z31+ self.QL[j] * z32)
        Z3 *= P
        '''Z4:制冷成本'''
        Z41 = 0
        Z42 = 0
        for k in range(1,self.vehicles + 1):
            if self.load[k] == 0:
                break
            for i in range(self.nodes):
                for j in range(1,self.nodes):
                    if (self.x[k][i][j] == 1):
                        Z41 += self.x[k][i][j] * (self.travel_time[j]+self.wait_time[j])
                        break
        Z41 *= Ce1/60
        for k in range(1,self.vehicles + 1):
            if self.load[k] == 0:
                break
            for j in range(1,self.nodes):
                if self.QL[j] > 0 :
                    Z42 += self.y[k][j] * self.service_time[j]
        Z42 *= Ce2/60
        Z4 = Z41 + Z42
        Z = Z1 + Z2 + Z3 + Z4
        return Z[0]
    
    def fitness_carb(self):
        '''Z5: 碳排放量'''
        e0 = 2.63
        w = 0.0066
        Q = 50
        p11 = 0.377
        p0 = 0.165
        Z5 = 0
        for k in range(1,self.vehicles + 1):
            if self.load[k] == 0:
                break
            for i in range(self.nodes):
                for j in range(self.nodes):
                    if (self.x[k][i][j] == 1):
                        Qij = self.QL[j] + self.demands[j]
                        p_Qij = p0 + (p11 - p0) * Qij / Q
                        Z5 += self.x[k][i][j] * self.distance[i][j] * \
                        (e0 * p_Qij + w * Qij)
                        break
        return Z5[0]
    
    def fitness_(self):
        self.x,self.y,self.load,flag = self.x_y()
        if (flag == 1):                 #车辆数b不大于可调度车辆数，为可行解
            fit_1 = self.fitness_time()
            fit_2 = self.fitness_cost()
            fit_3 = self.fitness_carb()
            return [fit_1,fit_2,fit_3]
        else:
            return [100000,100000,100000]                #车辆数大于可调度车辆数，为不可行解
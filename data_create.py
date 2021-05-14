# encoding: utf-8
import numpy as np
import solomon

def data_model():
    """Stores the data for the problem"""
    data = {}
    # Locations in block units
    df = solomon.solomon_data()
    _locations = df[:,1:3]
    demands = df[:,3:4]
    service_time = df[:,-1]
    numbers = df.shape[0] 
    capacities = np.ones((6,1))*40
    time_windows = df[:,4:6]

    # 构建数据集
    data["locations"] = _locations
    data["nodes"] = len(data["locations"])         # 需求节点数（含配送中心）
    data["depot"] = 0
    data["demands"] = demands
    data['mile_max'] = 100  # 每辆车的最大行驶里程数
    data["vehicle_capacities"] = 40  # 每辆车的最大容量
    data["vehicles"] = len(capacities)  # 供配送中心调配的车辆数
    data["time_windows"] = time_windows
    data["service_time"] = service_time  # 需求节点服务时间
    data['unit_trans_cost'] = 10  # 配送中心到需求点，车辆的单位运输成本，单位：元/公里
    data['vehicle_useCost'] = 300  # 车辆的使用成本，单位：元
    data['vehicle_speed'] = 40  # 配送中心到需求点，车辆的单位行驶速度，单位：公里/小时
    data['unit_ET_cost'] = 0.5  # 早到机会惩罚成本系数
    data['unit_LT_cost'] = 1.5  # 不满足时间窗要求的晚到单位惩罚成本
    data['unit_loss_cost'] = 0.5  # 单位货损成本系数
    data['extreme_position'] = [1,data["nodes"]-1]
    data['extreme_velocity'] = [-(data["nodes"]-2),(data["nodes"]-2)]
    data["distance"] = distance_callback(data["nodes"],data["locations"])
        
    return data

#计算两城市间的距离
def city_distance(city1,city2):
    dij = ( (float(city1[0] - city2[0]))**2 + (float(city1[1] - city2[1]))**2 )**0.5
    return dij

def distance_callback(nodes,locations):
    d = np.zeros((nodes,nodes))
    for i in range(nodes):
        for j in range(nodes):
            d[i][j] = city_distance(locations[i], locations[j])
    return d

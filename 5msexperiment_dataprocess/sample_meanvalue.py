'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script using for draw the relationship between average of RTT time and true distance in the 5ms indoor wireless experiment.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import random

zero_meantime = 20074.644
my_coefficient = 0.4

def read_data(distance):
    f = open ('5msexperiment\distance{}.txt'.format(distance), 'r')
    #f = open ('experiment2\distance{}.txt'.format(distance), 'r')

    data = f.readlines()

    time_data = []
    rssi_data = []
    for i in range(len(data)):
        data_split = data[i].split(' ')
        time_data.append(float(data_split[0]))
        rssi_data.append(float(data_split[1]))

    time_data = np.array(time_data)
    rssi_data = np.array(rssi_data)
    return time_data,rssi_data

distance_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

sample_times = 20
measurement_times = 200

def get_sample_mean(data,sample_times,measurement_times):
    sample_mean_list = []
    for i in range(sample_times):
        sample = sample = np.random.choice(data, size=measurement_times, replace=False)
        sample_mean_list.append(np.mean(sample))
    
    return sample_mean_list

distance_mean_list = []         #size = 16*20 each distance have 20 sample 
RSSI_mean_list = []
for i in distance_list:
    time_data, rssi_data = read_data(i)
    sample_distance_mean_list = get_sample_mean((time_data - zero_meantime)/(2*16000000)*299792458*my_coefficient, sample_times, measurement_times)
    sample_rssi_mean_list = get_sample_mean(rssi_data, sample_times, measurement_times)
    distance_mean_list.append(sample_distance_mean_list)
    RSSI_mean_list.append(sample_rssi_mean_list)


label_list = ['{} meters'.format(i) for i in range(16)]
plt.figure()
ax = plt.subplot(211)
ax.boxplot(distance_mean_list, labels= label_list, showfliers=False)
ax.set_ylabel('predict distance')
ax.grid()
ax.legend()

ax = plt.subplot(212)
ax.boxplot(RSSI_mean_list, labels= label_list, showfliers=False)
ax.set_ylabel('mean RSSI')
ax.grid()
ax.legend()
plt.show()





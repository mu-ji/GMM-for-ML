'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script using for draw the relationship between average of RTT time and true distance in the second wireless experiment.
'''
import numpy as np
import matplotlib.pyplot as plt
import os

zero_meantime = 20074.659

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

time_data_list = []
rssi_data_list = []
time_data_mean_list = []
time_data_var_list = []


for i in distance_list:
    time_data,rssi_data = read_data(i)
    time_data_list.append(time_data)
    rssi_data_list.append(rssi_data)

for i in time_data_list:
    time_data_mean_list.append(i.mean())
    time_data_var_list.append(i.var())

my_coefficient = 0.4
print(time_data_list)

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.scatter(distance_list,[(i-zero_meantime)/(2*16000000)*299792458*my_coefficient for i in time_data_mean_list],c = 'b', label = 'mean distance')
ax1.plot(distance_list,distance_list,c = 'r', label = 'true distance')
ax1.plot(distance_list,[i + 1 for i in distance_list],c = 'r', label = '+1 err', linestyle = '--')
ax1.plot(distance_list,[i - 1 for i in distance_list],c = 'r', label = '-1 err', linestyle = '--')
ax2 = fig.add_subplot(212)

for i in range(len(time_data_list)):
    for j in range(len(time_data_list[i])):
        time_data_list[i][j] -= zero_meantime
ax2.boxplot(time_data_list, showfliers=False)
plt.xticks(distance_list, ['{}'.format(i) for i in distance_list])
plt.legend()
plt.show()
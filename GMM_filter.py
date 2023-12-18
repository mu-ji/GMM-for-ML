'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script trying to using GMM filter on wireless experiment data
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import ml_experiment2 as ml
from sklearn.mixture import GaussianMixture

zero_meantime = 20074.659
my_coefficient = 0.4

sample_times = 10
distance_list = [i for i in range(16)]
data_list = []
train_data = []
test_data = []
for i in distance_list:
    data = ml.read_data(i)
    data_list.append(data)

train_data,test_data = ml.construct_train_and_test_data(data_list)
train_data = np.array(train_data)       #(16,2,1000)
test_data = np.array(test_data)         #(16,2,1000)

train_data_mean = np.mean(train_data, axis=2, keepdims=True)
test_data_mean = np.mean(test_data, axis=2, keepdims=True)

'''
plt.figure()
plt.hist(train_data[15][0].reshape((len(train_data[0][0]),1)), bins=30, density=True, alpha=0.5, label='Data')  # 数据直方图
plt.show()
'''

def GMM_filter(data):
    #best_aic,best_bic = compute_number_of_components(data,1,5)
    #n_components = best_aic  # 设置成分数量
    n_components = 2
    gmm = GaussianMixture(n_components=n_components)

    gmm.fit(data)
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_
    #coefficient = 0.9
    means = means[0][0]*weights[0] + means[1][0]*weights[1]
    #means = means[0][0]*coefficient + means[1][0]*(1-coefficient)
    return means,covariances,weights

def compute_number_of_components(data,min_components,max_components):
    # 计算AIC和BIC
    n_components_range = range(min_components, max_components+1)
    aic_values = []
    bic_values = []

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data)
        aic = gmm.aic(data)
        bic = gmm.bic(data)
        aic_values.append(aic)
        bic_values.append(bic)

    # 找到最小的AIC和BIC对应的成分数量
    best_aic = np.argmin(aic_values) + min_components
    best_bic = np.argmin(bic_values) + min_components
    return best_aic,best_bic

max_weight_mean_list = []
max_weight_covariance_list = []
for i in range(len(train_data)):
    max_weight_mean,max_weight_covariance,weights = GMM_filter(train_data[i][0].reshape((len(train_data[i][0]),1)))
    print(max_weight_mean)
    print(max_weight_covariance)
    max_weight_mean_list.append(max_weight_mean)
    max_weight_covariance_list.append(max_weight_covariance[0][0])

plt.figure()
ax = plt.subplot(211)
ax.plot([i for i in range(16)],train_data_mean[:,0:1,:].reshape((16,1)),c = 'r',label = 'time mean')
ax.plot([i for i in range(16)],max_weight_mean_list,c='b',label='filtered')
plt.legend()
ax = plt.subplot(212)
ax.scatter(distance_list,[(i-zero_meantime)/(2*16000000)*299792458*my_coefficient for i in max_weight_mean_list],c = 'b', label = 'mean distance')
ax.plot(distance_list,distance_list,c = 'r', label = 'true distance')
ax.plot(distance_list,[i + 1 for i in distance_list],c = 'r', label = '+1 err', linestyle = '--')
ax.plot(distance_list,[i - 1 for i in distance_list],c = 'r', label = '-1 err', linestyle = '--')
plt.legend()
plt.show() 
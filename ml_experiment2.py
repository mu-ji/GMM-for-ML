'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script trying to using ML to predict the distance based on the second RTT wireless experiment
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import joblib

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.mixture import GaussianMixture

import csv

def compute_err(distance_list, pre_list):
    error_list = []
    for i in range(len(distance_list)):
        error = np.abs(distance_list[i] - pre_list[i])
        error_list.append(error)
    
    return np.mean(error_list)

def read_data(distance):
    #f = open ('C:/Users/11422/Desktop/work_on_nrf52840/openwsn-fw/projects/nrf52840_dk/01bsp_rtt/experiment2/distance{}.txt'.format(distance), 'r')
    f = open ('indoor_without_people/distance{}.txt'.format(distance), 'r')
    time_list = []
    rssi_list = []
    data = f.readlines()
    #print(data)
    for i in range(len(data)):
        time_list.append(float(data[i].split(' ')[0]))
        rssi_list.append(float(data[i].split(' ')[1]))

    time = np.array(time_list)
    rssi = np.array(rssi_list)

    data = np.vstack((time,rssi))
    return data

def construct_train_and_test_data(data_list):
    train_data = []
    test_data = []
    for i in data_list:
        train_data.append(i[:,:60000])
        test_data.append(i[:,60000:])
    return train_data,test_data

def GMM_filter(data):
    #best_aic,best_bic = compute_number_of_components(data,1,5)
    #n_components = best_aic  # 设置成分数量
    n_components = 2
    gmm = GaussianMixture(n_components=n_components)
    try:
        gmm.fit(data)
    except:
        lenghts = len(data)
        gmm.fit(data.reshape((lenghts,1)))
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    return means,covariances,weights

def build_trainset_and_testset(train_data,test_data,n,p):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    train_x_time_component1_mean = []
    train_x_time_component1_var = []
    train_x_time_component2_mean = []
    train_x_time_component2_var = []
    train_x_time_component1_weight = []
    train_x_time_component2_weight = []
    train_x_rssi_mean = []
    train_x_rssi_var = []
    train_x_time_mean = []
    train_x_time_var = []

    test_x_time_component1_mean = []
    test_x_time_component1_var = []
    test_x_time_component2_mean = []
    test_x_time_component2_var = []
    test_x_time_component1_weight = []
    test_x_time_component2_weight = []
    test_x_rssi_mean = []
    test_x_rssi_var = []
    test_x_time_mean = []
    test_x_time_var = []

    for i in range(len(train_data)):        #each distance
        for j in range(n):                  #sample numbers
            k = np.random.randint(60000 - p)        #sample length
            train_x.append(train_data[i,:,k:k+p])
            train_y.append(i)

            means,covariances,weights = GMM_filter(train_data[i,0,k:k+p])
            train_x_time_component1_mean.append(means[0][0])
            train_x_time_component1_var.append(covariances[0][0][0])
            train_x_time_component2_mean.append(means[1][0])
            train_x_time_component2_var.append(covariances[1][0][0])
            train_x_time_component1_weight.append(weights[0])
            train_x_time_component2_weight.append(weights[1])

            train_x_rssi_mean.append(np.mean(train_data[i,1,k:k+p]))
            train_x_rssi_var.append(np.var(train_data[i,1,k:k+p]))
            train_x_time_mean.append(np.mean(train_data[i,0,k:k+p]))
            train_x_time_var.append(np.var(train_data[i,0,k:k+p]))

        k = np.random.randint(60000 - p)
        test_x.append(test_data[i,:,k:k+p])
        test_y.append(i)

        means,covariances,weights = GMM_filter(test_data[i,0,k:k+p])
        test_x_time_component1_mean.append(means[0][0])
        test_x_time_component1_var.append(covariances[0][0][0])
        test_x_time_component2_mean.append(means[1][0])
        test_x_time_component2_var.append(covariances[1][0][0])
        test_x_time_component1_weight.append(weights[0])
        test_x_time_component2_weight.append(weights[1])

        test_x_rssi_mean.append(np.mean(test_data[i,1,k:k+p]))
        test_x_rssi_var.append(np.var(test_data[i,1,k:k+p]))
        test_x_time_mean.append(np.mean(test_data[i,0,k:k+p]))
        test_x_time_var.append(np.var(test_data[i,0,k:k+p]))

        train_packet = [train_x,train_y,
                        train_x_time_component1_mean,train_x_time_component1_var,
                        train_x_time_component2_mean,train_x_time_component2_var,
                        train_x_time_component1_weight,train_x_time_component2_weight,
                        train_x_rssi_mean,train_x_rssi_var,
                        train_x_time_mean,train_x_time_var]
        
        test_packet = [test_x,test_y,
                        test_x_time_component1_mean,test_x_time_component1_var,
                        test_x_time_component2_mean,test_x_time_component2_var,
                        test_x_time_component1_weight,test_x_time_component2_weight,
                        test_x_rssi_mean,test_x_rssi_var,
                        test_x_time_mean,test_x_time_var]
    
    return train_packet,test_packet
'''
def data_loss(X,Y,output):
    data_loss = 0
    for i in range(len(Y)):
        data_loss = data_loss + (output[i]*2*16000000/299792458 + 20074.659 - X[i][0])**2

    return data_loss
'''


def main():
    sample_times = 1000
    distance_list = [i for i in range(1,12)]
    data_list = []
    train_data = []
    test_data = []
    for i in distance_list:
        data = read_data(i)
        data_list.append(data)

    train_data,test_data = construct_train_and_test_data(data_list)
    train_dataset = np.array(train_data)       #(16,2,1000)
    train_dataset[:,0:1,:] -= 20074.659
    test_dataset = np.array(test_data)         #(16,2,1000)
    test_dataset[:,0:1,:] -= 20074.659

    train_packet,test_packet = build_trainset_and_testset(train_dataset,test_dataset,sample_times,200)

    train_x = np.array(train_packet[0])
    train_y = np.array(train_packet[1])

    train_x_time_component1_mean = np.array(train_packet[2])
    train_x_time_component1_var = np.array(train_packet[3])
    train_x_time_component2_mean = np.array(train_packet[4])
    train_x_time_component2_var = np.array(train_packet[5])
    train_x_time_component1_weight = np.array(train_packet[6])
    train_x_time_component2_weight = np.array(train_packet[7])

    train_x_rssi_mean = np.array(train_packet[8])
    train_x_rssi_var = np.array(train_packet[9])
    train_x_time_mean = np.array(train_packet[10])
    train_x_time_var = np.array(train_packet[11])

    test_x = np.array(test_packet[0])
    test_y = np.array(test_packet[1])

    test_x_time_component1_mean = np.array(test_packet[2])
    test_x_time_component1_var = np.array(test_packet[3])
    test_x_time_component2_mean = np.array(test_packet[4])
    test_x_time_component2_var = np.array(test_packet[5])
    test_x_time_component1_weight = np.array(test_packet[6])
    test_x_time_component2_weight = np.array(test_packet[7])

    test_x_rssi_mean = np.array(test_packet[8])
    test_x_rssi_var = np.array(test_packet[9])
    test_x_time_mean = np.array(test_packet[10])
    test_x_time_var = np.array(test_packet[11])

    train_merged_array = np.concatenate((train_x_time_component1_mean[:, np.newaxis],
                                         train_x_time_component1_var[:, np.newaxis],
                                         train_x_time_component2_mean[:, np.newaxis],
                                         train_x_time_component2_var[:, np.newaxis],
                                         train_x_time_component1_weight[:, np.newaxis],
                                         train_x_time_component2_weight[:, np.newaxis],
                                         train_x_rssi_mean[:, np.newaxis],
                                         train_x_rssi_var[:, np.newaxis],
                                         train_x_time_mean[:, np.newaxis],
                                         train_x_time_var[:, np.newaxis]), axis=1)

    X = train_merged_array[:,:]
    print(X.shape)
    Y = train_y

    lin_model = LinearRegression()
    lin_model.fit(X, Y)

    ridge_model = Ridge()
    ridge_model.fit(X,Y)

    lasso_model = Lasso()
    lasso_model.fit(X,Y)

    test_merged_array = np.concatenate((test_x_time_component1_mean[:, np.newaxis],
                                         test_x_time_component1_var[:, np.newaxis],
                                         test_x_time_component2_mean[:, np.newaxis],
                                         test_x_time_component2_var[:, np.newaxis],
                                         test_x_time_component1_weight[:, np.newaxis],
                                         test_x_time_component2_weight[:, np.newaxis],
                                         test_x_rssi_mean[:, np.newaxis],
                                         test_x_rssi_var[:, np.newaxis],
                                         test_x_time_mean[:, np.newaxis],
                                         test_x_time_var[:, np.newaxis]), axis=1)

    test_x = test_merged_array[:,:]


    def save_data_to_csv(data, filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    
    save_data_to_csv([list(row) for row in X.tolist()],'train_x.csv')
    save_data_to_csv([list(row) for row in Y.reshape((len(Y),1)).tolist()],'train_y.csv')
    save_data_to_csv([list(row) for row in test_x.tolist()],'test_x.csv')
    save_data_to_csv([list(row) for row in test_y.reshape((len(test_y),1)).tolist()],'test_y.csv')

    ridge_predictions = ridge_model.predict(test_x)
    lasso_predictions = lasso_model.predict(test_x)

    ridge_mse = mean_squared_error(test_y, ridge_predictions)
    lasso_mse = mean_squared_error(test_y, lasso_predictions)
    print('ridge mse loss:',ridge_mse)
    print('lasso mse loss:',lasso_mse)
    
    plt.figure()
    plt.scatter([i for i in range(len(test_x))],ridge_predictions, c = 'b',label = 'ridge_predictions')
    plt.scatter([i for i in range(len(test_x))],lasso_predictions, c = 'y',label = 'lasso_predictions')
    plt.plot([i for i in range(len(test_x))],[i for i in range(len(test_x))],c = 'r',label = 'true value')
    plt.title('ridge and lasso')
    plt.legend()
    plt.show()
    
    X = torch.from_numpy(X[:,:]).float()

    Y = torch.from_numpy(Y.reshape(11*sample_times,1)).float()
    test_x = torch.from_numpy(test_x[:,:]).float()
    test_y = torch.from_numpy(test_y.reshape((11,1))).float()
    
    class RegressionNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RegressionNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.elu = nn.ELU()
            self.fc3 = nn.Linear(hidden_size, output_size)



        def forward(self, x): 
            out = self.fc1(x)
            out = self.elu(out)
            out = self.fc2(out)
            out = self.elu(out)
            out = self.fc3(out)
            return out
        
    input_size = 10
    hidden_size = 128
    output_size = 1

    model = RegressionNet(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):

        outputs = model(X)
        loss = criterion(outputs, Y) #+ 0.0001*data_loss(X,Y,outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_loss.append(loss.item())

        predictions = model(test_x)
        tt_loss = criterion(predictions,test_y)
        test_loss.append(tt_loss.item())

    with torch.no_grad():
        predictions = model(test_x)

    tt_loss = criterion(predictions,test_y)
    print(tt_loss.item())
    torch.save(model.state_dict(), 'without_people_model')

    predictions = predictions.numpy()

    #with_people_model = RegressionNet(input_size, hidden_size, output_size)
    #with_people_model.load_state_dict(torch.load('with_people_model'))

    #with torch.no_grad():
    #    outdoor_predictions = outdoor_model(test_x)
    
    #outdoor_loss = criterion(outdoor_predictions,test_y)

    #new_model = joblib.load('without_people_model')

    #new_model_pre = with_people_model(test_x)
    #new_model_pre = new_model_pre.detach().numpy()

    #multi_pre = []
    #for i in range(len(distance_list)):
    #    #print(predictions[i][0])
    #    multi_pre.append((predictions[i][0]+new_model_pre[i])/2)


    plt.figure()
    ax = plt.subplot(211)
    ax.scatter(distance_list,predictions, c = 'b',label = 'ML_predictions error={}'.format(compute_err(distance_list,predictions.reshape(11,1))))
    #ax.scatter(distance_list,new_model_pre, c = 'g',label = 'with people situation error={}'.format(compute_err(distance_list,new_model_pre)))
    #ax.scatter(distance_list,multi_pre, c = 'y',label = 'multi_predictions error={}'.format(compute_err(distance_list,multi_pre)))
    ax.plot(distance_list,distance_list,c = 'r',label = 'true value')
    #ax.scatter(distance_list,[(i)/(2*16000000)*299792458*0.4 for i in test_x_time_mean],c = 'y', label = 'only RTT time(coefficient=0.4)')
    ax.plot(distance_list,[i+1 for i in distance_list],c = 'r',label = '+1 error boundary',linestyle = '--')
    ax.plot(distance_list,[i-1 for i in distance_list],c = 'r',label = '-1 error boundary',linestyle = '--')
    ax.set_xlabel('true distance')
    ax.set_ylabel('predict distance')
    ax.set_title('without people walking situation')
    ax.legend()
    ax = plt.subplot(212)
    ax.plot([i for i in range(num_epochs)],train_loss,c='r',label='training loss')
    ax.plot([i for i in range(num_epochs)],test_loss,c='b',label='test loss')
    ax.legend()
    plt.show()
    
    print(predictions)
    #print(new_model_pre)

    

main()

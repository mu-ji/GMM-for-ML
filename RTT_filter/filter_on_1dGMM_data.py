import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


LoS_weight = 0.7
NLoS_weight = 1 - LoS_weight

true_data = []
for i in range(1000):
    true_data.append(LoS_weight*np.random.normal(loc=1, scale=0.04) + NLoS_weight*np.random.normal(loc=1.5, scale=0.5))
true_data = np.array(true_data)


n_components = 2
gmm = GaussianMixture(n_components=n_components)
gmm.means_init = np.array([1,2]).reshape((2,1))
gmm.covariances_init = np.array([0.04,0.5]).reshape((2,1))
gmm.fit(true_data.reshape(1000,1))

means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

print(means)
print(covariances)
print(weights)

# 可视化生成的数据
plt.hist(true_data, bins=50, density=True, alpha=0.7)
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.title('Generated GMM Data')
plt.show()
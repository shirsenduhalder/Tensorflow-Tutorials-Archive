import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
num_sanples = 100000
prob_density = 0
mean = np.array([0, 0])
mean1 = np.matrix(mean)
cov = np.array([[1, 0.7], [0.7, 1]])
cov1 = np.matrix(cov)

x_list, y_list = [], []
accepted_samples_count = 0
normalizer = np.sqrt(((2*np.pi)**2)*np.linalg.det(cov))

x_initial, y_initial = 0, 0
x1, y1 = x_initial, y_initial

for i in range(num_sanples):
	mean_trans = np.array([x1, y1])
	cov_trans = np.array([[0.2, 0], [0, 0.2]])
	x2, y2 = np.random.multivariate_normal(mean_trans, cov_trans).T
	X = np.array([x2, y2])
	X2 = np.matrix(X)
	X1 = np.matrix(mean_trans)

	mahalobnis_dist2 = (X2 - mean1)*np.linalg.inv(cov)*(X2 - mean1).T
	prob_density2 = (1/float(normalizer))*np.exp(-0.5*mahalobnis_dist2)
	mahalobnis_dist1 = (X1 - mean1)*np.linalg.inv(cov)*(X1 - mean1).T
	prob_density1 = (1/float(normalizer))*np.exp(-0.5*mahalobnis_dist1)

	acceptance_ratio = prob_density2[0, 0]/float(prob_density1[0, 0])

	if((acceptance_ratio >= 1) | ((acceptance_ratio < 1) and (acceptance_ratio >= np.random.uniform(0, 1)))):
		x_list.append(x2)
		y_list.append(y2)

		x1 = x2
		y1 = y2

		accepted_samples_count += 1

end_time = time.time()
print("Time taken to sample {} samples: {}".format(accepted_samples_count, (end_time - start_time)))
print("Acceptance Ratio : {}".format(accepted_samples_count/num_sanples))
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x_list, y_list, color='black')
print("Mean of Sampled points")
print(np.mean(x_list), np.mean(y_list))
print("Covariance of Sampled points")
print(np.cov(x_list, y_list))
plt.show()
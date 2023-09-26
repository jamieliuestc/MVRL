import numpy as np
import torch
import random

import utils
import math
from sklearn.cluster import KMeans
from sklearn import preprocessing		
from mvlearn.cluster import MultiviewKMeans

					
def clustering(buffer, policy, kinds_number, km_num, ind, sample_rate):

	temp_buffer = buffer.sampleall()

	temp0 = temp_buffer[0]
	temp1 = temp_buffer[1]
	temp2 = temp_buffer[2]
	temp3 = temp_buffer[3]
	temp4 = temp_buffer[4]
	temp5 = temp_buffer[5]

	temp_buffer0 = policy.extract_state(temp0)
	temp_buffer1 = policy.extract_cur_state(temp1, temp2)
	temp_buffer2 = policy.extract_nr_state(temp4,temp5)	
	print(temp_buffer0.size())
	print(temp_buffer1.size())
	print(temp_buffer2.size())
	
	temp_buffer0 = temp_buffer0.cpu().detach().numpy()
	temp_buffer1 = temp_buffer1.cpu().detach().numpy()
	temp_buffer2 = temp_buffer2.cpu().detach().numpy()
	clus_buffer = [temp_buffer0,temp_buffer1,temp_buffer2]


	m_kmeans = MultiviewKMeans(n_clusters=km_num, n_init=20, random_state=0)
	labels = m_kmeans.fit_predict(clus_buffer)


	labels = np.array(labels)

	index = buffer.Choose_sample(labels)
	kinds_i = [0]
	sample_number = int((len(labels) // km_num) * sample_rate)

	for i in range(len(index)):
		kinds_i.append(kinds_i[i] + len(index[i]) - 1)
		if len(index[i]) > sample_number:
			index_i = random.sample(range(len(index[i])), sample_number)
			index[i] = index[i][index_i]
		ind = np.hstack((ind,index[i]))

	kinds_number.append(kinds_i)

	print(ind.shape)
	print(kinds_i)
	print(kinds_number)

	return kinds_number, ind

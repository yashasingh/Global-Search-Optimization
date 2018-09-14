import numpy as np
import pandas as pd

# class DataSetModel(object):
# 	def __init__(self):
# 		self.data = pd.read_csv("OVA_Lung.arff")
# 		self.tissue_col = data.Tissue.tolist()
# 		self.df = data.drop('Tissue', axis=1)
# 		self.train_list, self.validation_list, self.test_list = [], [], []
#         self.train_label, self.validation_label, self.test_label = [], [], []
#         self.count_other, self.count_lung = 0, 0

#     def create_dataset(self):
#     	for ind,row in df.iterrows():
# 			if data['Tissue'][ind] == 'Other':
# 				if count_other <= 0.6*1419:
# 					train_list.append(np.array(row.tolist())
# 					train_label.append(tissue_col[ind])
# 					count_other+=1
# 				else if count_other > 0.6*1419 and count_other <= 0.8*1419:
# 					validation_list.append(np.array(row.tolist()))
# 					validation_label.append(tissue_col[ind])
# 					count_other+=1
# 				else:
# 					test_list.append(np.array(np.array(row.tolist()))
# 					test_label.append(tissue_col[ind])
# 			else:
# 				if count_lung <= 0.6*126:
# 					train_list.append(np.array(row.tolist()))
# 					train_label.append(tissue_col[ind])
# 					count_lung+=1
# 				else if count_lung > 0.6*126 and count_lung <= 0.8*126:
# 					validation_list.append(np.array(np.array(row.tolist)))
# 					validation_label.append(tissue_col[ind])
# 					count_lung+=1
# 				else:
# 					test_list.append(np.array(row.tolist()))
# 					test_label.append(tissue_col[ind])





data = pd.read_csv("OVA_Lung.arff")
tissue_col = data.Tissue.tolist()
# tissue_col.tolist()
df = data.drop('Tissue', axis=1)
train_list, validation_list, test_list = [], [], []
train_label, validation_label, test_label = [], [], []
count_other, count_lung = 0, 0
for ind,row in df.iterrows():
	if data['Tissue'][ind] == 'Other':
		if count_other <= 0.6*1419:
			train_list.append(np.array(row))
			train_label.append(np.array([0.0, 1.0]))
			count_other+=1
		elif count_other > 0.6*1419 and count_other <= 0.8*1419:
			validation_list.append(np.array(row))
			validation_label.append(np.array([0.0, 1.0]))
			count_other+=1
		else:
			test_list.append(np.array(row))
			test_label.append(np.array([0.0, 1.0]))
	else:
		if count_lung <= 0.6*126:
			train_list.append(np.array(row))
			train_label.append(np.array([1.0, 0.0]))
			count_lung+=1
		elif count_lung > 0.6*126 and count_lung <= 0.8*126:
			validation_list.append(np.array(np.array(row)))
			validation_label.append(np.array([1.0, 0.0]))
			count_lung+=1
		else:
			test_list.append(np.array(row))
			test_label.append(np.array([1.0, 0.0]))

train_list = np.array(train_list).astype(float)
train_label = np.array(train_label).astype(float)

validation_list = np.array(validation_list).astype(float)
validation_label = np.array(validation_label).astype(float)

test_list = np.array(test_list).astype(float)
test_label = np.array(test_label).astype(float)


# print(train_list)





# example = {'field1':[1,2,3,4,5,6,7], 'field2':['a','b','c','d','e','f','g'], 'label':[0,1,0,1,1,0,0]}
# data_frame = pd.DataFrame(data=example)
# data_frame.to_csv('file.csv')

# data = pd.read_csv("OVA_Lung.arff")
# count = 0
# for ind,row in data.iterrows():
# 	if count <= 5:
# 		print(row.toList())
# 		count += 1


# #
# 1 - I need dictionary 
# so convert  the arff to dict format 
# so that row1 values are key and corresponding is column
# 2 - divide data into 3 csv
# numpy array
# train labels, validation, test labels - 6 lists 
# else if 0.6, 0.2, 0.2
# #
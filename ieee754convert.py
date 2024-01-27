import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import struct
df = pd.read_csv('/opt/data/SWaT数据集/data_newlabel.csv', header=None)
df_inputs = df.iloc[:, :-1]
df_labels = df.iloc[:, -1]
def float32_to_int32(value):
    # 使用struct模块将float32转换为bytes对象
    bytes_representation = struct.pack('!f', value)
    bytes = [byte for byte in bytes_representation]
    # print(bytes)
    return np.array(bytes)
np_inputs = df_inputs.values
# np_inputs = np.random.rand(3,4)
np_inputs = np.expand_dims(np_inputs,2)
# print(np_inputs)
np_inputs = np.apply_along_axis(float32_to_int32, axis=2, arr=np_inputs)
# print(np_inputs)
np.save('ieee754_inputs.npy', np_inputs)
np.save('labels.npy', df_labels)
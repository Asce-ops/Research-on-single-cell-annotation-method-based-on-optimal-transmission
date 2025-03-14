# /data_ssd/ysQiu/anaconda3/envs/scDOT/bin/python
import time
start: float = time.time()
import os
import pandas as pd
import numpy as np
from scDOT import scDOT

os.chdir(path='/data_ssd/ysQiu/TanWeiqiang/scDOT')

batch = pd.read_csv(filepath_or_buffer="./datasets/pbmc/pbmc/batch.csv", header=0, index_col=False).x.values
query_name = batch[0]
ref_name = batch[1:] # 全部读取

M, x, y_hat, score, y_hat_with_unseen = scDOT(loc="./datasets/pbmc/pbmc/", ref_name=ref_name, query_name=query_name)

file_name = "./datasets/pbmc/pbmc/" + f"{query_name}_label.csv"
y_true = pd.read_csv(filepath_or_buffer=file_name, header=0, index_col=0).iloc[:,0].values
accuracy1 = np.sum(y_hat == y_true) / y_true.shape[0]
accuracy2 = np.sum(y_hat_with_unseen == y_true) / y_true.shape[0]
print(f"The annotation accuracy of {query_name} is {accuracy1, accuracy2}")
end: float = time.time()
print('累计耗时', end-start)
# /data_ssd/ysQiu/anaconda3/envs/scDOT/bin/python

import numpy as np
import pandas as pd
import functools
from collections import Counter
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor
import math
import ot
from utils import pp, concatenate, find_index, vstack, cal_mmd, dist_ensemble, H
import time


def cal_ct_margin(ref_label_list: list[np.ndarray[str]], label_set: list[str]) -> np.ndarray:
    '''标签的边缘分布'''
    ct_margin: list = list()
    for r in range(len(ref_label_list)):
        s_f: np.ndarray = np.zeros(shape=len(label_set)) # 一维数组，代表第 r 个参考数据集中各个标签出现的概率分布
        sum_feat: pd.core.frame.DataFrame = pd.DataFrame(data=Counter(ref_label_list[r]).items()) # 统计参考数据集中每个标签出现的次数并将其转化为pd.DataFrame；第一列是标签，第二类是该标签出现的次数
        sum_feat = sum_feat.sort_values(by=0) # 将标签出现的次数按照标签排序
        id: np.ndarray[int] = find_index(a=label_set, b=sum_feat.iloc[:,0])
        s_f[id] = sum_feat.iloc[:,1] / np.sum(a=sum_feat.iloc[:,1])
        ct_margin.append(s_f)

    ct_margin: np.ndarray = functools.reduce(vstack, ct_margin) # 垂直堆叠各个参考数据集标签的分布
    counts: np.ndarray = ct_margin.astype(bool).sum(axis=0) # 一位数组，每个元素对应相应标签在多少个查询数据集中出现的概率非0
    ct_margin = np.sum(a=ct_margin, axis=0) / counts # 标签在各个参考数据集中出现的平均概率（不考虑没有该标签的数据集）
    ct_margin[np.isnan(ct_margin)] = 0 # 用 0 填充缺失值
    ct_margin = ct_margin / np.sum(a=ct_margin) # 一维数组归一化
    
    return ct_margin

def cal_single_dist(ref_dat: np.ndarray, ref_label: np.ndarray[str], query_dat: np.ndarray) -> np.ndarray:
    '''
    ref_dat: 单个参考数据集的特征
    ref_label: 单个参考数据集的标签
    '''
    
    ct_count: pd.core.frame.DataFrame = pd.DataFrame(data=Counter(ref_label).items()) # 统计该参考数据集中每个标签出现的次数并将其转化为pd.DataFrame；第一列是标签，第二类是该标签出现的次数
    ct_count = ct_count.sort_values(by=0)
    
    def compute_distance(i) -> np.ndarray:
        '''计算查询数据集中每个细胞与该参考数据集中属于第 i 个标签的细胞的特征平均表达水平的余弦距离'''
        id: np.ndarray = np.where(ref_label == ct_count.iloc[i, 0])[0] # 用于定位到第 i 个标签的细胞的特征
        ct_feat: np.ndarray = np.mean(a=np.array(object=ref_dat[id, :]), axis=0).reshape(1, ref_dat.shape[1]) # 该数据集中所有标签为第 i 个标签的细胞的特征的均值（二位数组）
        return distance.cdist(XA=query_dat, XB=ct_feat, metric='cosine').T # XA中每一行与XB的余弦距离（因为是多行一列，所以转置为一行多列）

    with ThreadPoolExecutor() as executor: # 多线程计算
        dist_list: list[np.ndarray] = list(executor.map(compute_distance, range(ct_count.shape[0])))

    # '''该方案已废弃'''
    # ref_label: list[str] = list(ref_label)
    # n: int = len(ref_label)
    # for i in range(ct_count.shape[0]):
    #     w: float = ref_label.count(ct_count.iloc[i, 0]) / n
    #     dist_list[i] *= w

    dist: np.ndarray = np.vstack(tup=dist_list).T # 垂直堆叠多个余弦距离数组再转置

    print('-------------------------------')
    for i in dist_list:
        print('单个标签的运输成本矩阵形状', i.shape)

    return dist


def cal_multi_dist(ref_dat: list[np.ndarray], ref_label: list[np.ndarray[str]], label_set: list[str], query_dat: np.ndarray) -> list[pd.core.frame.DataFrame]:
    '''
    计算初始的运输成本矩阵
    形状：查询数据集的细胞个数 * 参考数据集标签类别数
    '''
    
    dist: list[np.ndarray] = list()
    for i in range(len(ref_dat)):
        single_dist: np.ndarray = cal_single_dist(ref_dat=ref_dat[i], ref_label=ref_label[i], query_dat=query_dat)
        
        temp_dist: np.ndarray = np.full(shape=[query_dat.shape[0], len(label_set)], fill_value=0.)
        '''将 single_dist 的各列赋给 temp_dist 的指定列'''
        id: np.ndarray[int] = find_index(a=label_set, b=ref_label[i])
        temp_dist[:, id] = single_dist
        
        temp_dist /= temp_dist.max()
        temp_dist: pd.core.frame.DataFrame = pd.DataFrame(data=temp_dist, columns=label_set, index=np.array(object=np.arange(0, temp_dist.shape[0], 1, dtype='int'), dtype='object'))
        
        dist.append(temp_dist)
    
    for i in range(len(dist)):
        print(f'第{i+1}个运输成本矩阵的形状为：{dist[i].shape}')
    
    return dist


def D_inpute_with_expr(Dist: list[pd.core.frame.DataFrame], label_set: np.ndarray[str], dat_list: list[np.ndarray], label_list: list[np.ndarray[str]]) -> tuple[list[pd.core.frame.DataFrame], np.ndarray]:
    '''
    dat_list: 预处理后参考数据集的特征
    label_list: 参考数据集上的标签（不去重）
    label_set: 参考数据集上的标签（去重）
    Dist: 运输成本矩阵
    '''
    c: int = len(dat_list)
    '''所有参考数据集上公共的标签'''
    intersection: set[str] = set(label_list[0])
    for df in label_list[1:]:
        ct = set(df)
        intersection = intersection.intersection(ct)
    
    common_ct: np.ndarray[str] = np.array(object=list(intersection)) # 所有参考数据集上公共的标签
    # 运输成本矩阵（只考虑公共标签）
    D_temp: list[pd.core.frame.DataFrame] = list()
    for d in Dist:
        D_temp.append(d[common_ct])
    # 从参考数据集的特征集中剔除掉属于非公共标签的细胞
    dat_temp: list[np.ndarray] = list()
    for d in range(len(dat_list)):
        idx = np.isin(label_list[d], common_ct)
        dat_temp.append(dat_list[d][idx])
    
    w_list: list[float] = list()
    for l in range(c):
        idx_dat: list[int] = ([ii for ii in range(len(dat_temp)) if ii != l])
        for ll in idx_dat:
            W: float = cal_mmd(X=dat_temp[l], Y=dat_temp[ll]) # 计算两个数据集 X 和 Y 之间的最大均值差异（Maximum Mean Discrepancy, MMD）的平方
            W: float = math.exp(-W)
            w_list.append(W)
    
    w_list: np.ndarray = np.array(object=w_list).reshape(len(dat_list), len(dat_list)-1)
    w_list_norm = w_list / np.sum(a=w_list, axis=1)[:, np.newaxis]
    
    '''处理运输成本矩阵'''
    D_inpute: list[pd.core.frame.DataFrame] = list()
    for l in range(c):
        ind: pd.core.series.Series = Dist[l].eq(0).all() # 每一列是否全为0
        if np.sum(ind) == 0: # 没有全为 0 的列
            D_inpute.append(Dist[l])
        else:
            index = [ii for ii in range(c) if ii != l]
            D_unique = list()
            for i in range(np.sum(ind)):
                a = [Dist[ii][label_set[ind][i]] for ii in index]
                D_unique.append(pd.concat(a, axis=1))
            temp = list()
            for i in range(np.sum(ind)):
                id_temp = D_unique[i].ne(0).any()
                w_use = w_list_norm[l,:][id_temp]
                if w_use.all() == 0:
                    temp.append(pd.Series(np.ones(Dist[l].shape[0])))
                else:
                    w_use = w_use/ np.sum(w_use)
                    temp.append(D_unique[i].iloc[:,np.array(object=range(D_unique[i].shape[1]))[id_temp]] @ w_use)
            inputed = pd.concat(temp, axis=1)
            Dist[l][label_set[ind]] = inputed
            D_inpute.append(Dist[l])
            
    return D_inpute, w_list


def init_solve(D: list[np.ndarray], a: np.ndarray, b: np.ndarray, lambda1: float, w_piror: np.ndarray | None = None, lambda2: float = 0.01) -> tuple[float, np.ndarray]:
    '''更新初始的权重和并得到权重熵项的系数'''
    
    n, t = D[0].shape # 运输成本矩阵的形状

    # '''修改（原版将这部分注掉）'''
    # new_D: list[np.ndarray] = list()
    # for i in D:
    #     for j in range(t):
    #         tmp: np.ndarray = np.zeros(shape=(n, t))
    #         tmp[:, j] = i[:, j]
    #         new_D.append(tmp)
    # D = new_D
    
    c: int = len(D)
    
    if w_piror is None:
        # w_piror = np.full(shape=(c, 1), fill_value=1 / c)
        w_piror = np.full(shape=(c), fill_value=1 / c)
    else:
        w_piror = w_piror.copy()
        
    M: np.ndarray = dist_ensemble(dist=D, weight=w_piror) # 运输成本矩阵的加权和
    x: np.ndarray = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, lambda1, lambda2) # 概率矩阵
    
    '''权重熵项的系数'''
    # tt: list[float] = [(np.dot(D[l], x.T).trace()) for l in range(c)]
    tt = [x.T[l % t, :] @ D[l][:, l % t] for l in range(c)]
    print('tt', tt)
    beta: float = np.median(a=tt) # 确定权重熵项的系数
    
    '''更新权重'''
    numerator = [math.exp(-1 * tt[l] / beta) for l in range(c)]
    w = np.array(object=[numerator[l] / np.sum(a=numerator) for l in range(c)])
    print('更新初始的权重', w)
    
    return beta, w

def _solve(D: list[np.ndarray], a: np.ndarray, b: np.ndarray, w_piror: np.ndarray | None = None, lambda1: float = 0.01, lambda2: float = 0.01, beta: float = 1, maxIter: int = 100, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:

    n, t = D[0].shape

    # '''修改（原版将这部分注掉）'''
    # new_D: list[np.ndarray] = list()
    # for i in D:
    #     for j in range(t):
    #         tmp: np.ndarray = np.zeros(shape=(n, t))
    #         tmp[:, j] = i[:, j]
    #         new_D.append(tmp)
    # D = new_D

    c = len(D)
    
    if w_piror is None:
        # w_prev = np.full(shape=(c, 1), fill_value=1 / c)
        w_prev = np.full(shape=(c), fill_value=1 / c)
    else:
        w_prev = w_piror.copy()
        
    loss_prev = 1e4

    # maxIter = 1 # 迭代次数超过1则权重会滑向一个非常奇怪的结果（原版注掉）
    for i in range(maxIter):
        
        M = dist_ensemble(dist=D, weight=w_prev) # 运输成本矩阵的加权和
        # M = M / np.max(a=M) # 多余，dist_ensemble 函数返回的已经是归一化后的结果
        x: np.ndarray = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, lambda1, lambda2) # 概率矩阵
        
        '''更新权重'''
        # tt = [(np.dot(D[l], x.T).trace()) for l in range(c)]
        tt = [x.T[l % t, :] @ D[l][:, l % t] for l in range(c)]
        print('tt', tt)
        numerator = [math.exp(-1 * np.dot(D[l], x.T).trace() / beta) for l in range(c)]
        w = np.array(object=[numerator[l] / np.sum(numerator) for l in range(c)])
        print('新一轮的权重为：', w)
        
        '''计算损失函数'''
        loss1 = 0
        for l1, l2 in zip(tt, w):
            loss1 += l1*l2
        loss2 = lambda1 * H(x) + lambda2 * np.sum(a * np.log1p(a / (np.sum(x, axis=1) + 1e-10))) + lambda2 * np.sum(b * np.log1p(b / (np.sum(x, axis=0) + 1e-10)))
        loss3 = beta*H(w)
        loss = loss1 + loss2 + loss3
        
        diff_loss = np.absolute(loss_prev - loss)/np.absolute(loss_prev + 1e-10)
        print('损失函数值：', loss, 'loss1：', loss1, 'loss2：', loss2, 'loss3：', loss3)
        
        if diff_loss < tol:
            break
        
        w_prev = w.copy()
        loss_prev = loss.copy()

    print(f'一共迭代了{i+1}次')
    return x, w, M


def solve(D: list[np.ndarray], a: np.ndarray, b: np.ndarray, label_set: list[str], beta: float, w_piror: np.ndarray | None = None, lambda1: float = 0.01, lambda2: float = 0.01, maxIter2: int = 100, tol: float = 1e-4):
    
    
    x, w, M = _solve(D=D, a=a, b=b, w_piror=w_piror, maxIter=maxIter2, lambda1=lambda1, lambda2=lambda2, beta=beta, tol=tol)
    
    y_hat = np.argmax(a=x, axis=1) # 每一行最大值所在的索引
    y_hat = np.array(object=y_hat, dtype='object')
    ct: list[str] = sorted(label_set)
    for i in range(len(ct)):
        y_hat[np.where(y_hat == i)[0]] = ct[i]


    return x, w, y_hat, M

def scDOT(loc: str, ref_name: np.ndarray[object], query_name: str, lambda1: float = 0.01, lambda2: float = 0.01, threshold: float = 0.9):
    '''
    Input:
    :loc: Where reference data and query data are stored.
    :ref_name: A list of names of the reference datasets.
    :query_name: The name of the query datasets.
    :lambda1: Numerical value, and the default value is 0.01.
    :lambda2: Numerical value, and the default value is 0.01.
    :threshold: The threshold for unseen cell type identification, and the default value is 0.9.
    Output:
    final_annotation: 1D-array, final annotation including unseen cell-type identification.
    m: 1D-array, metric for unseen cell-type identification.
    '''
    # threshold: float = 0.95
    start_time: float = time.time()
    expression: list[pd.core.frame.DataFrame] = list() # 参考数据集和查询数据集的特征
    label_s: list[np.ndarray[str]] = list() # 参考数据集的标签
    for i in ref_name:
        file_name: str = loc + f"{i}_cell.csv" # 特征
        a: pd.core.frame.DataFrame = pd.read_csv(filepath_or_buffer=file_name, header=0, index_col=0)
        expression.append(a)
        file_name: str = loc + f"{i}_label.csv" # 标签
        a: np.ndarray = pd.read_csv(filepath_or_buffer=file_name, header=0, index_col=0).iloc[:,0].values
        label_s.append(a)
    file_name: str = loc + f"{query_name}_cell.csv" # 查询数据集特征
    expression_t: pd.core.frame.DataFrame = pd.read_csv(filepath_or_buffer=file_name, header=0, index_col=0)
    expression.append(expression_t)
    del file_name, a, i
    end_time: float = time.time()
    time_used: float = end_time - start_time
    print(f"Time for data loading is {time_used}")

    b_hat: np.ndarray = np.array(object=np.sum(a=expression_t, axis=1)) / np.sum(a=np.sum(a=expression_t)) # 按行求和再除以总和（因为expression_t 是 DataFrame，所以 np.sum(a=expression_t) 返回的是 Series）
    
    expression: list[np.ndarray] = pp(dat_list=expression) # 特征预处理
    expression_t: np.ndarray = expression[-1] # 预处理后查询数据集的特征
    expression: list[np.ndarray] = expression[0: -1] # 预处理后参考数据集的特征
    
    label_set: np.ndarray[str] = functools.reduce(concatenate, label_s) # 水平堆叠各个参考数据集的标签
    label_set: list[str] = sorted(np.unique(ar=label_set)) # 各参考数据集的标签去重后排序转化为列表
    a_hat: np.ndarray = cal_ct_margin(ref_label_list=label_s, label_set=label_set) # 标签的边缘分布
    D: list[pd.core.frame.DataFrame] = cal_multi_dist(ref_dat=expression, ref_label=label_s, label_set=label_set, query_dat=expression_t) # 运输成本矩阵
    # D, _ = D_inpute_with_expr(Dist=D, label_set=np.array(object=label_set), dat_list=expression, label_list=label_s) # 处理后的运输成本矩阵
    D: list[np.ndarray] = [np.array(object=ll) for ll in D] # 将各个参考数据集对应的运输成本矩阵转换为 numpy 数组
    
    # w: np.ndarray = np.full(shape=(len(D), 1), fill_value=1 / len(D)) # 初始化权重
    lambda3, w = init_solve(D=D, a=b_hat, b=a_hat, lambda1=lambda1, lambda2=lambda2)
    x, w, y_hat, M = solve(
        D=D, a=b_hat, b=a_hat, label_set=label_set, 
        # w_piror=w, 
        lambda1=lambda1, lambda2=lambda2, beta=lambda3
    )
    
    x = x / np.sum(a=x, axis=1, keepdims=True) # 逐行归一化，转换为真正的概率矩阵
    score = np.max(a=x, axis=1)
    
    y_hat_with_unseen = y_hat.copy()
    y_hat_with_unseen[np.where(score < threshold)[0]] = "unseen"
    
    print(f'查询数据集的形状为{expression_t.shape}')
    print('标签个数', len(label_set))
    print(f'概率矩阵的形状{x.shape}')
    return M, x, y_hat, score, y_hat_with_unseen
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDOneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr

from sklearn.neighbors import NearestNeighbors

plt.rcParams['font.sans-serif'] = ['SimHei']

# 得到训练和验证集
def get_train_valid_q1():
    root_path = '/data1/wyy/projects/_competition/math_model_v2/Data'
    excel_name1 = '1.xlsx'
    excel_name2 = '2.xlsx'
    excel_name3 = '3.xlsx'
    excel_name_time = 'additional.xlsx'

    excel_answer = '4.xlsx'
    
    # 读取Excel文件
    table1 = pd.read_excel(os.path.join(root_path, excel_name1))
    table2 = pd.read_excel(os.path.join(root_path, excel_name2))
    table3 = pd.read_excel(os.path.join(root_path, excel_name3))
    table_time = pd.read_excel(os.path.join(root_path, excel_name_time))
    table_answer = pd.read_excel(os.path.join(root_path, excel_answer))
    
    all_Base_info = np.array([[]], dtype=np.float64)
    
    # print(table1['年龄'])
    # print(table1['年龄'])
    # series = table1.set_index(['年龄'], drop=True)
    # plt.figure(figsize=(10, 6))
    # series['脑出血前mRS评分'].plot()
    # plt.show()
    
    # 遍历所有160个病人
    for i in range(160):
        info_dict = {}
        x = np.array([], dtype=np.float64)
        row1 = table1.loc[i]
        row2 = table2.loc[i]
        
        # 遍历每个病人的第index=4列到index=19列
        for j in range(4, 4+19):
            if j == 2:
                continue
            elif j == 5: # 对应性别列
                # 男为0，女为1
                sex = 0 if row1[j] == '男' else 1
                x = np.append(x, sex)
            elif j==15:  # 对应血压列
                # 获得血压数值，存储为列表格式[]
                xueya = np.array([eval(k) for k in row1[j].split('/')])
                x = np.append(x, xueya)
            else:
                # 将其他列对应的数值添加到x中
                x = np.append(x, row1[j])
                
        # 将x
        x = x[np.newaxis]

        if all_Base_info.size == 0:
            all_Base_info = x  # 将其中对应的x传递给all_Base_info
        else:
            all_Base_info=np.concatenate((all_Base_info, x), axis=0)
    
    
    # 将table1中的性别列从字符串转换为0或1
    table1['性别'] = table1['性别'].map({'男':1, '女':0})
    
    # 选择的特征列
    info_select = ['年龄', '性别', '高血压病史', '卒中病史', '糖尿病史']
    
    # 得到表格中对应上述选择特征列的信息
    all_Base_info = table1[info_select].values.astype(np.float64)
    
    # 得到表格2中对应的所有的HM体积信息
    all_HM_info = table2['HM_volume'].values.astype(np.float64).reshape(-1, 1)
    
    # 首次检查对应的影像检查结果
    # index from 3 to 23
    for i in range(3, 24):
        tem_data = table2[table2.columns[i]].values.astype(np.float64).reshape(-1, 1)
        # 拼接数据
        all_HM_info = np.concatenate((all_HM_info, tem_data), axis=1)
        # break
    
    print(all_HM_info.shape)

    first_check_seq = table1['入院首次影像检查流水号'].values
    first_check_seq_2 = table2['首次检查流水号'].values
    first_check_seq_3 = table3['流水号'].values
    first_check_seq_time = table_time['入院首次检查流水号'].values
    
    sub_name = table2['ID']
    print("异常数据 in table1, 2\n 流水号不对应:\n", sub_name[first_check_seq!=first_check_seq_2])
    print("异常数据 in table1, time\n 流水号不对应:\n", sub_name[first_check_seq!=first_check_seq_time]) 
    print("异常数据 in table2, time\n 流水号不对应:\n", sub_name[first_check_seq_2!=first_check_seq_time]) 
    print("异常数据: table1与table2=time的流水号存在不对应的情况")

    shape_query_buf = []
    table12_query_buf = []

    # table12_test_query_buf
    for seq in first_check_seq:
        # print(np.where(first_check_seq_3==seq)[0])
        if (np.where(first_check_seq_3==seq)[0]).size == 0:
            shape_query_buf.append(None)
            print(np.where(first_check_seq_3==seq), seq, sub_name[first_check_seq_2==seq])
        else:
            # 得到index索引值在100以内或大于等于132的
            if np.where(first_check_seq==seq)[0][0]<100 or np.where(first_check_seq==seq)[0][0]>=132:
                table12_query_buf.append(np.where(first_check_seq==seq)[0][0])
                shape_query_buf.append(np.where(first_check_seq_3==seq)[0][0])
    
    # X = np.concatenate((X, HM_info), axis=1)

    all_shape_info = table3[table3.columns[2]].values.astype(np.float64).reshape(-1, 1)
    
    for i in range(3, 33):
        # print(table3.columns[i])
        tem_data = table3[table3.columns[i]].values.astype(np.float64).reshape(-1, 1)
        all_shape_info = np.concatenate((all_shape_info, tem_data), axis=1)

    
    shape_info = all_shape_info[shape_query_buf]
    Base_info = all_Base_info[table12_query_buf]
    
    # HM_info = all_HM_info[table12_query_buf]
    print(shape_info.shape)
    
    # X = np.concatenate((X, HM_info), axis=1)
    # X = np.concatenate((X, shape_info), axis=1)
    
    # 构建X和Y作为train和valid数据
    X = Base_info
    # X = np.concatenate((X, HM_info), axis=1)
    # X = np.concatenate((X, shape_info), axis=1)
    # Y = table_answer['是否发生血肿扩张']
    Y = table_answer['问题1：血肿扩张'].values[2:].astype(np.float64)[table12_query_buf]

    seq = []
    p_value_buf = []
    
    for i in range(X.shape[1]):
        pearson_corr, p_value = pearsonr(X[:, i], Y)
        
        if np.isnan(p_value):
            p_value = 0.
        p_value_buf.append(p_value)
        # print(p_value)
        if p_value>0.1:
            seq.append(i)

    p_value_buf = np.array(p_value_buf)
    normalize = plt.Normalize(vmin=p_value_buf.min(), vmax=p_value_buf.max())

    # cmap = plt.cm.get_cmap('viridis')
    # # 将浮点数映射到RGB颜色空间
    # colors = cmap(normalize(p_value_buf))
    # print(colors)
    # colors[seq, 1] = 0
    # colors[seq, 0] = 1.
    # colors[seq, 2] = 0
    # plt.figure()

    # # 添加标题和坐标轴标签
    # plt.title("Pearson 相关系数")
    # plt.xlabel("Features")
    # plt.ylabel("Pearson correlation coefficient")

    # plt.show()
    # plt.bar(np.arange(len(p_value_buf)), p_value_buf, color=colors)
    # plt.savefig("Pearson.jpg")
    # plt.show()


    # X = X[:, seq]
    means = np.mean(X, axis=0)
    # # print(means)
    # stds = np.std(X, axis=0)

    X = (X-means)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # X = X[:, stds!=0]
    # X = np.random.random((100, 49))
    # Y = np.random.randint(0, 2, (100,))
    scaler = scaler.fit(X)

    # print(stds!=0)
    X_train = X[:100]
    y_train = Y[:100]

    X_valid = X[100:]
    y_valid = Y[100:]
   
    # X_train, X_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    # print(X_train[:, 3])

    # for j in range(72):
    #     print(j)
    #     plt.plot(X_train[:, j])
    #     plt.show()

    # print(X_train[0])
    print("X_train: {}, X_valid: {}, y_train: {}, y_valid: {}".format(X_train, X_valid, y_train, y_valid))
    return X_train, X_valid, y_train, y_valid

if __name__=='__main__':
    get_train_valid_q1()
#     return

# print(X_train.shape)

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Predict on validation set
# y_pred = model.predict(X_valid)

# # Calculate accuracy
# accuracy = accuracy_score(y_valid, y_pred)

# print(accuracy)

# svm = SGDOneClassSVM(max_iter=1000)
# svm.fit(X_train, y_train)
# y_pred_svm = svm.predict(X_valid)

# accuracy_svm = accuracy_score(y_valid, y_pred_svm)
# print(accuracy_svm)


    
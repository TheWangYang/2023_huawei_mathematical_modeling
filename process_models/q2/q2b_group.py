import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture


# 定义存放数据的路径
root_path = "/data1/wyy/projects/_competition/mathmodel_code/data"

# 定义表格名称
table1_name = "1.xlsx"
table2_name = "2.xlsx"
table3_name = "3.xlsx"
# 4-答案表
table4_name = "4.xlsx"
table_additional = "additional.xlsx"


# 读取Excel文件
table1 = pd.read_excel(os.path.join(root_path, table1_name))
table2 = pd.read_excel(os.path.join(root_path, table2_name))
table3 = pd.read_excel(os.path.join(root_path, table3_name))
table_addi = pd.read_excel(os.path.join(root_path, table_additional))


# 根据现有的index遍历获得同一个亚组的病人的水肿体积随时间变化情况
def get_X_y(index_keep):
    # 创建需要返回的X和y
    X = []
    y = []
    
    # 根据表2中的每个检查点对应的ED_volume以及（该检查点对应的时间-首次检查点对应的时间+发病到首次检查点对应的时间）
    for i in index_keep:
        # 获得表格中每行数据
        table1_row = table1.loc[i]
        table2_row = table2.loc[i]
        # print("ID: {}".format(table2_row[0]))
        
        table3_row = table3.loc[i]
        table_addi_row = table_addi.loc[i]
                
        # 先得到表2中对应的ED_volume中，判断是否为空，不为空得到对应的检查点到发病的时间间隔
        # table2_ED_volume_index = [13, 36, 59, 82, 105, 128, 151, 174, 197]
        # 只记录患者从发病到首次检查时间
        table2_ED_volume_index = [13]
        
        # 遍历其中的每个ED_volume_index得到y并计算对应的检查点时间间隔为x
        for index in table2_ED_volume_index:
            # 判断是否为空，不为空进入逻辑
            if not np.isnan(table2_row[index]):
                # 打印table2_row[index]数据类型
                # print(type(table2_row[index]))
                # 根据此时的检查点计算从发病到此时间点的间隔
                curr_check_number = table2_row[index - 12]
                # 根据流水号在附表中找到对应流水号在附表中的列index
                # 如果没有找到表示出现错误，直接跳过即可
                if (np.where(curr_check_number == table_addi_row)[0]).size != 0:
                    check_number_addi_index = np.where(curr_check_number == table_addi_row)[0][0]
                    # print("check_number_addi_index: {}".format(check_number_addi_index))
                    # 得到检查流水号对应的时间点列索引index
                    time_for_check_number_addi_index = check_number_addi_index - 1
                    # 然后计算当前时间点距离首次检查时间点的小时数
                    interval_to_first_time = (table_addi_row[time_for_check_number_addi_index] - table_addi_row[2]).total_seconds() / 3600
                    # 然后，从表1中获得从发病到首次检查的时间间隔
                    first_check_interval_to_onset = table1_row[14]
                    # 然后得到从当前检查点时间距离发病的时间间隔
                    curr_interval = interval_to_first_time + first_check_interval_to_onset
                    # 将当前检查点对应的时间间隔作为x轴，添加到X列表中
                    # print("curr_interval: {}".format(curr_interval))
                    X.append(curr_interval)
                    # 将该值添加到y列表中
                    y.append(table2_row[index])
        
    # 打印X和y
    print("X size: {}".format(len(X)))
    print("y size: {}".format(len(y)))
    
    return X, y

# 使用曲线拟合
def regression_linear(label, index_keep, X, y):
    
    print(np.max(X))
    print(np.max(y))
    
    # ------------------------------------使用高斯拟合曲线-------------------------------------
    # 转换为NumPy数组
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # 高斯混合模型拟合
    gmm = GaussianMixture(n_components=2)  # 可根据需要调整混合成分数
    gmm.fit(X, y)

    # # 生成拟合曲线的X值
    X_fit = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
    y_fit = gmm.predict(X_fit)

    # 绘图
    plt.scatter(X, y, label='Original Data')
    plt.plot(X_fit, y_fit, color='red', label='Fitted Curve (Gaussian Mixture)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    # 保存的时候使用label作为区分
    plt.savefig("process_models/q2/output/q2b_regression_gaussian_{}.png".format(label))
    # 清除画布
    plt.clf()
    
    # ----------------------------------使用多项式拟合曲线--------------------------------------
    # 多项式回归拟合
    degree = 3  # 多项式阶数，可以根据需要调整
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)

    # 生成拟合曲线的X值
    X_fit = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
    y_fit = model.predict(X_fit)

    # 绘图
    plt.scatter(X, y, label='Original Data')
    plt.plot(X_fit, y_fit, color='red', label=f'Fitted Curve (degree={degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    # 使用label区分是哪个类别
    plt.savefig("process_models/q2/output/q2b_regression_polynomial_{}.png".format(label))
    # 清除画布
    plt.clf()
    
    # ------------------------------------得到预测结果-----------------------------------------
    # 得到从发病到首次检查时间点的时间间隔，得到X轴表示
    # 根据表2中的每个检查点对应的ED_volume以及（该检查点对应的时间-首次检查点对应的时间+发病到首次检查点对应的时间）
    X_new = []
    y_new = []
    
    # 遍历index_keep
    for i in index_keep:
        # 获得表格中每行数据
        table1_row = table1.loc[i]
        table2_row = table2.loc[i]
        
        X_new.append(table1_row[14])
        y_new.append(table2_row[13])
    
    X_new = np.array(X_new).reshape(-1, 1)
    # 使用高斯曲线预测
    # y_pred = gmm.predict(X_new)
    # 使用多项式预测
    y_pred = model.predict(X_new)
    
    # # 计算拟合曲线和真实值之间的残差
    residuals = np.abs(y_new - y_pred)
    
    print(y_new, y_pred, residuals)
    
    # 填充表4答案表格中对应的残差列
    # pd.set_option('display.float_format', lambda x: '%.3f' % x)
    table4_answer = pd.read_excel(os.path.join(root_path, table4_name))
    
    # print(table4_answer)
    
    # 使用 iloc 给特定列赋值
    index_keep_plus_2 = [x + 2 for x in index_keep]
    table4_answer.iloc[index_keep_plus_2, 6] = residuals
    label_list = [label] * len(index_keep)
    table4_answer.iloc[index_keep_plus_2, 7] = label_list
    table4_answer.to_excel(os.path.join(root_path, table4_name), index=False, float_format='%.3f')
    
    print("success")
    return


'''
聚类函数
其中K为聚类数量为存放数据的路径地址
'''

def cluster_K_group():
    # 创建需要返回的X
    X = []

    # 得到表1中病人对应的特征
    for i in range(100):  # You can use range(100) directly
        # 获得表格中每行数据
        table1_row = table1.loc[i]

        # 提取对应的特征列
        feature1 = np.array([table1_row[1]])  # 对应90天mRS
        feature2 = np.array([table1_row[4]])

        # 处理性别特征
        feature_age = np.array([1]) if table1_row[5] == '男' else np.array([0])

        feature3 = table1_row[6:15].values.flatten()

        # 处理血压
        xueya = np.array([eval(k) for k in table1_row[15].split('/')])

        feature5 = table1_row[16:23].values.flatten()

        # 合并特征
        combined_feature = np.concatenate((feature1, feature2, feature_age, feature3, xueya, feature5))
        X.append(combined_feature)

    # 转换为NumPy数组
    X = np.array(X)

    # 使用KMeans进行聚类
    n_clusters = min(5, len(X))  # 不超过5个簇
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    # 获取簇标签
    # 对应100前100个病人
    cluster_labels = kmeans.labels_
    print("cluster_labels: {}".format(cluster_labels))
    print("cluster_labels.size: {}".format(cluster_labels.size))
    label_counts = pd.Series(cluster_labels).value_counts()
    print("label_counts: {}".format(label_counts))
    
    
    # 建立保存对应的5个类别[0, 1, 2, 3, 4]对应的病人在表1中的index
    label0_keep = []
    label1_keep = []
    label2_keep = []
    label3_keep = []
    label4_keep = []
    
    # 根据标签，将前100个病人进行分组
    # 共分为5个类别
    for i, label in enumerate(cluster_labels):
        # 得到表格的第i行
        table1_row = table1.loc[i]
        
        if label == 0:
            label0_keep.append(i)
        elif label == 1:
            label1_keep.append(i)
        elif label == 2:
            label2_keep.append(i)
        elif label == 3:
            label3_keep.append(i)
        elif label == 4:
            label4_keep.append(i)
        
    # 对于每个保存的keep列表使用与2a中相同的方法进行回归预测
    X,y = get_X_y(label0_keep)
    regression_linear(0, label0_keep, X, y)
    
    X,y = get_X_y(label1_keep)
    regression_linear(1, label1_keep, X, y)
    
    X,y = get_X_y(label2_keep)
    regression_linear(2, label2_keep, X, y)
    
    X,y = get_X_y(label3_keep)
    regression_linear(3, label3_keep, X, y)
    
    X,y = get_X_y(label4_keep)
    regression_linear(4, label4_keep, X, y)
    
    print("-----------------success---------------")
    

# 单元测试代码
if __name__ == "__main__":
    cluster_K_group()
    
    
    



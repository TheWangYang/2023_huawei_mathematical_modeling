import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
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


'''
得到X轴和y轴需要的数据
root_path为存放数据的路径地址
'''
def get_X_y():
    # 创建需要返回的X和y
    X = []
    y = []
    
    # 根据表2中的每个检查点对应的ED_volume以及（该检查点对应的时间-首次检查点对应的时间+发病到首次检查点对应的时间）
    for i in range(0, 100):
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
def regression_linear(X, y):
    
    
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
    plt.savefig("process_models/q2/output/q2a_regression_gaussian.png")
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
    plt.savefig("process_models/q2/output/q2a_regression_polynomial.png")
    
    # ------------------------------------得到预测结果-----------------------------------------
    # 得到从发病到首次检查时间点的时间间隔，得到X轴表示
    # 根据表2中的每个检查点对应的ED_volume以及（该检查点对应的时间-首次检查点对应的时间+发病到首次检查点对应的时间）
    X_new = []
    y_new = []
    
    for i in range(0, 100):
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
    table4_answer.iloc[2: 102, 5] = residuals
    table4_answer.to_excel(os.path.join(root_path, table4_name), index=False, float_format='%.3f')
    
    print("success")
    return
    
    
# 单元测试代码
if __name__ == "__main__":
    X, y = get_X_y()
    regression_linear(X, y)
    
    
    
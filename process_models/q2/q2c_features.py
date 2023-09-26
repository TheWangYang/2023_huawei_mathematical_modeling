import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



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


def features_impact_ED_volume():
    # 准备存储特征和目标数据的列表
    all_features = []
    all_targets = []

    # 分析前100个病人
    for i in range(100):
        # 获得表格中每行数据
        table1_row = table1.loc[i]
        table2_row = table2.loc[i]
        
        table3_row = table3.loc[i]
        table_addi_row = table_addi.loc[i]
                
        # 得到当前第一次复检点与首次检查得到的ED_volume的差值，获得增量
        curr_to_first_check_interval_ED_volume = table2_row[36] - table2_row[13]
        
        # 提取16-22列特征
        features = table1_row[16:23].values

        # 存储特征和目标数据
        all_features.append(features)
        all_targets.append(curr_to_first_check_interval_ED_volume)

    # 合并特征和目标变量为一个数据集
    features_array = np.array(all_features)
    targets_array = np.array(all_targets)

    # 构建决策树模型
    model = DecisionTreeRegressor()
    model.fit(features_array, targets_array)

    # 获取特征重要性
    feature_importances = model.feature_importances_
    
    print("feature_importances: {}".format(feature_importances))
    
    # 中文显示设置
    fontprop = fm.FontProperties(fname='/data1/wyy/projects/_competition/mathmodel_code/utils/fonts/TIMESBD.TTF')  # 替换成你的字体文件路径
    plt.rcParams['font.family'] = fontprop.get_name()
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    # plt.bar(['脑室引流', '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经'], feature_importances)
    plt.bar(['0', '1', '2', '3', '4', '5', '6'], feature_importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance from Decision Tree')
    plt.xticks(rotation=45)  # 旋转x轴标签，以避免重叠
    plt.savefig("process_models/q2/output/q2c_features_decision_tree.png")
    plt.clf()
    
    print("--------------------success-------------------")

# 调用函数进行特征影响分析
features_impact_ED_volume()

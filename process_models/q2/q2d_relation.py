import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 题目要求：分析血肿体积，水肿体积，和治疗方法三者之间的关系
# 思路：首先得到（第一复检点的血肿体积-首次检查对应的血肿体积），（第一复检点-首次检查对应的水肿体积），（获得治疗方法特征）
# 然后使用关系分析方法来分析三者之间的关系

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

"""
定义函数：用于分析三者之间的关系
"""
def relationship_between_three():
    # 准备存储特征和目标数据的列表
    all_features = []
    all_HM_targets = []
    all_ED_targets = []
    
    # 分析前100个病人
    for i in range(100):
        # 获得表格中每行数据
        table1_row = table1.loc[i]
        table2_row = table2.loc[i]
        
        table3_row = table3.loc[i]
        table_addi_row = table_addi.loc[i]
                
        # 得到当前第一次复检点与首次检查得到的ED_volume的差值，获得增量
        curr_to_first_check_interval_ED_volume = table2_row[36] - table2_row[13]
        
        # 得到当前第一次复检点与首次检查得到的HM_volume的差值，获得增量
        curr_to_first_check_interval_HM_volume = table2_row[25] - table2_row[2]
        
        # 提取16-22列特征
        features = table1_row[16:23].values

        # 存储特征和目标数据
        all_features.append(features)
        all_HM_targets.append(curr_to_first_check_interval_HM_volume)
        all_ED_targets.append(curr_to_first_check_interval_ED_volume)
        

    # 合并特征和目标变量为一个数据集
    # 修正可能导致错误的代码
    features_array = np.array(all_features).astype('float16')
    HM_targets_array = np.array(all_HM_targets).astype('float16')
    ED_targets_array = np.array(all_ED_targets).astype('float16')
    
    
    # 计算HM目标与特征的相关系数
    hm_correlations = np.corrcoef(features_array.T, HM_targets_array)[0, 1:]

    # 计算ED目标与特征的相关系数
    ed_correlations = np.corrcoef(features_array.T, ED_targets_array)[0, 1:]

    # 打印相关系数
    print("Correlation between features and HM_targets:", hm_correlations)
    print("Correlation between features and ED_targets:", ed_correlations)

    # 绘制散点图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar([0, 1, 2, 3, 4, 5, 6], hm_correlations, alpha=0.5)
    plt.xlabel('Features')
    plt.ylabel('HM Targets')
    plt.title('Scatter Plot: Features vs HM Targets')

    plt.subplot(1, 2, 2)
    plt.bar([0, 1, 2, 3, 4, 5, 6], ed_correlations, alpha=0.5)
    plt.xlabel('Features')
    plt.ylabel('ED Targets')
    plt.title('Scatter Plot: Features vs ED Targets')
    plt.tight_layout()
    plt.savefig("process_models/q2/output/q2d_relation.png")

    print("--------------------success-------------------")


if __name__ == "__main__":
    relationship_between_three()







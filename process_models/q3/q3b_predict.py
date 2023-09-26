import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 题目要求：结合多个特征对mRS进行预测
# 思路：保存前100个病人的特征，然后作为X，同时以mRS为目标值


# 定义存放数据的路径
root_path = "/data1/wyy/projects/_competition/mathmodel_code/data"

# 定义表格名称
table1_name = "1.xlsx"
table2_name = "2.xlsx"
table3_name = "3.xlsx"
# 4-答案表
table4_name = "4.xlsx"
table_additional = "additional.xlsx"


# Read Excel files and handle empty values
table1 = pd.read_excel(os.path.join(root_path, table1_name))
table2 = pd.read_excel(os.path.join(root_path, table2_name))
table3 = pd.ExcelFile(os.path.join(root_path, table3_name))

table3_ED = None
table3_HM = None

# 遍历表格3中每个工作表
for sheet_name in table3.sheet_names:
    # 在这里可以对每个工作表的 DataFrame 进行相应的操作
    # 例如，打印工作表名称和 DataFrame 的头部
    print(f'工作表名称: {sheet_name}')
    
    if sheet_name == "ED":
        table3_ED = pd.read_excel(table3, "ED")
    elif sheet_name == "Hemo":
        table3_HM = pd.read_excel(table3, "Hemo")

# 读取附加表
table_addi = pd.read_excel(os.path.join(root_path, table_additional))


# 预测mRS
def get_X_y_for_mRS(k):
    
    # 定义保存前100个病人对应的mRS数值
    mRS_targets = []
    
    # 汇总特征列
    all_features = []
    
    # 分析前100个病人
    for i in range(k):
        
        # 获得表格中每行数据
        table1_row = table1.loc[i]
        table2_row = table2.loc[i]
        table_addi_row = table_addi.loc[i]
        
        # ---------------------------------处理表格1----------------------------------
        # 年龄列
        age_feature = table1_row[4]
        
        # 性别列
        sex_feature = 1 if table1_row[5] == '男' else 0
        
        # 提取6-14列特征
        feature1 = table1_row[6:15].values
        
        # 血压列
        xueya = np.array([eval(k) for k in table1_row[15].split('/')])
        
        # 提取表格table1中的16-22列特征
        feature2 = table1_row[16:23].values
        
        # ---------------------------------处理表格2----------------------------------
        # 得到table2中对应首次检查结果列
        feature3 = table2_row[2: 24].values
        
        # ---------------------------------处理表格3----------------------------------
        # 得到ED和HM子表数据
        target_ED_row_data = table3_ED.loc[table3_ED['流水号'] == table1_row[3]]
        target_HM_row_data = table3_HM.loc[table3_HM['流水号'] == table1_row[3]]
                
        # 获取整行数据
        if len(target_ED_row_data) > 0 and len(target_HM_row_data) > 0:
            # 得到对应ED的特征列
            feature4 = target_ED_row_data[2: 33].values.flatten()
            # 得到对应HM的特征列
            feature5 = target_HM_row_data[2: 33].values.flatten()
            
            # print("Row data for the target value {}: {}".format(table1_row[3], target_row_data))
        else:
            # 如果不存在，那么直接跳过
            print("Target value {} not found in the specified column.".format(table1_row[3]))
        
        
        # --------------------------拼接其他随访时间表2和表3对饮的特征列-----------------------------
        # -----------------------------第k次随访对应的表2中特征列-----------------------------------
        # 处理随访特征中的空值，将NaN替换为0
        table2_suifang1_feature = np.nan_to_num(table2_row[25: 47].values)
        table2_suifang2_feature = np.nan_to_num(table2_row[48: 70].values)
        table2_suifang3_feature = np.nan_to_num(table2_row[71: 93].values)
        table2_suifang4_feature = np.nan_to_num(table2_row[94: 116].values)
        table2_suifang5_feature = np.nan_to_num(table2_row[117: 139].values)
        table2_suifang6_feature = np.nan_to_num(table2_row[140: 162].values)
        table2_suifang7_feature = np.nan_to_num(table2_row[163: 185].values)
        table2_suifang8_feature = np.nan_to_num(table2_row[186: 208].values)
        
        
        # ---------------------------------第k次随访对应的表3中ED特征列----------------------------------
        table3_ED_suifang1_row_data = table3_ED.loc[table3_ED['流水号'] == table2_row[24]]
        table3_ED_suifang2_row_data = table3_ED.loc[table3_ED['流水号'] == table2_row[47]]
        table3_ED_suifang3_row_data = table3_ED.loc[table3_ED['流水号'] == table2_row[70]]
        table3_ED_suifang4_row_data = table3_ED.loc[table3_ED['流水号'] == table2_row[93]]
        table3_ED_suifang5_row_data = table3_ED.loc[table3_ED['流水号'] == table2_row[116]]
        table3_ED_suifang6_row_data = table3_ED.loc[table3_ED['流水号'] == table2_row[139]]
        table3_ED_suifang7_row_data = table3_ED.loc[table3_ED['流水号'] == table2_row[162]]
        table3_ED_suifang8_row_data = table3_ED.loc[table3_ED['流水号'] == table2_row[185]]
        
        # 处理表3中ED和HM特征列，将NaN替换为0
        table3_ED_suifang1_feature = np.nan_to_num(table3_ED_suifang1_row_data[2: 33].values.flatten())
        table3_ED_suifang2_feature = np.nan_to_num(table3_ED_suifang2_row_data[2: 33].values.flatten())
        table3_ED_suifang3_feature = np.nan_to_num(table3_ED_suifang3_row_data[2: 33].values.flatten())
        table3_ED_suifang4_feature = np.nan_to_num(table3_ED_suifang4_row_data[2: 33].values.flatten())
        table3_ED_suifang5_feature = np.nan_to_num(table3_ED_suifang5_row_data[2: 33].values.flatten())
        table3_ED_suifang6_feature = np.nan_to_num(table3_ED_suifang6_row_data[2: 33].values.flatten())
        table3_ED_suifang7_feature = np.nan_to_num(table3_ED_suifang7_row_data[2: 33].values.flatten())
        table3_ED_suifang8_feature = np.nan_to_num(table3_ED_suifang8_row_data[2: 33].values.flatten())
        
        
        # -----------------------------第k次随访对应的表3中ED特征列-----------------------------------------
        table3_HM_suifang1_row_data = table3_HM.loc[table3_HM['流水号'] == table2_row[24]]
        table3_HM_suifang2_row_data = table3_HM.loc[table3_HM['流水号'] == table2_row[47]]
        table3_HM_suifang3_row_data = table3_HM.loc[table3_HM['流水号'] == table2_row[70]]
        table3_HM_suifang4_row_data = table3_HM.loc[table3_HM['流水号'] == table2_row[93]]
        table3_HM_suifang5_row_data = table3_HM.loc[table3_HM['流水号'] == table2_row[116]]
        table3_HM_suifang6_row_data = table3_HM.loc[table3_HM['流水号'] == table2_row[139]]
        table3_HM_suifang7_row_data = table3_HM.loc[table3_HM['流水号'] == table2_row[162]]
        table3_HM_suifang8_row_data = table3_HM.loc[table3_HM['流水号'] == table2_row[185]]

        # 处理表3中HM特征列，将NaN替换为0
        table3_HM_suifang1_feature = np.nan_to_num(table3_HM_suifang1_row_data[2: 33].values.flatten())
        table3_HM_suifang2_feature = np.nan_to_num(table3_HM_suifang2_row_data[2: 33].values.flatten())
        table3_HM_suifang3_feature = np.nan_to_num(table3_HM_suifang3_row_data[2: 33].values.flatten())
        table3_HM_suifang4_feature = np.nan_to_num(table3_HM_suifang4_row_data[2: 33].values.flatten())
        table3_HM_suifang5_feature = np.nan_to_num(table3_HM_suifang5_row_data[2: 33].values.flatten())
        table3_HM_suifang6_feature = np.nan_to_num(table3_HM_suifang6_row_data[2: 33].values.flatten())
        table3_HM_suifang7_feature = np.nan_to_num(table3_HM_suifang7_row_data[2: 33].values.flatten())
        table3_HM_suifang8_feature = np.nan_to_num(table3_HM_suifang8_row_data[2: 33].values.flatten())
        
        
        # 汇总X特征列
        # 将零维数组扩展为一维数组
        age_feature = np.expand_dims(age_feature, axis=0)
        # all_feature = np.concatenate((age_feature, [sex_feature], feature1, xueya, feature2, feature3, feature4, feature5))
        
        # 合并所有特征
        all_feature = np.concatenate((age_feature, np.array([sex_feature]), feature1, xueya, feature2, feature3, feature4, feature5, 
                                    table3_ED_suifang1_feature, table3_ED_suifang2_feature,
                                    table3_ED_suifang3_feature, table3_ED_suifang4_feature,
                                    table3_ED_suifang5_feature, table3_ED_suifang6_feature,
                                    table3_ED_suifang7_feature, table3_ED_suifang8_feature,
                                    table3_HM_suifang1_feature, table3_HM_suifang2_feature,
                                    table3_HM_suifang3_feature, table3_HM_suifang4_feature,
                                    table3_HM_suifang5_feature, table3_HM_suifang6_feature,
                                    table3_HM_suifang7_feature, table3_HM_suifang8_feature,
                                    table2_suifang1_feature, table2_suifang2_feature,
                                    table2_suifang3_feature, table2_suifang4_feature,
                                    table2_suifang5_feature, table2_suifang6_feature,
                                    table2_suifang7_feature, table2_suifang8_feature))

        all_features.append(all_feature)
        
        # 添加y目标值列
        # 得到mRS目标值作为y值
        mRS_targets.append(table1_row[1])
    
    
    return all_features, mRS_targets


# 预测mRS函数
def predict_mRS():
    # ------------------------------得到X和y之后，使用模型预测--------------------------
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # -------------------------首先得到前100个病人的数据，用于训练模型------------------
    X, y = get_X_y_for_mRS(100)
    
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    X[np.isnan(X)] = 0
    
    print(X.shape)
    print(y.shape)

    # 创建决策树模型
    model = DecisionTreeClassifier(random_state=42)
    
    # 训练模型
    model.fit(X, y)

    # 在训练集上进行预测（可以修改为测试集）
    y_pred = model.predict(X)

    # 计算准确率（因为使用的是训练数据，这里计算的是训练准确率，实际应用时应该使用独立的测试集）
    accuracy = accuracy_score(y, y_pred)
    print('模型准确率为:', accuracy)
    
    # # 绘图
    # plt.scatter(X, y, label='Original Data')
    # plt.plot(X, y_pred, color='red', label=f'Fitted Curve (degree={degree})')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.legend()
    # plt.savefig("process_models/q3/output/q3a_predict.png")
    # plt.clf()
    
    # -------------------------使用所有160个病人的数据来进行预测得到mRS-----------------------
    # 得到所有病人的数据
    X_all, y_all = get_X_y_for_mRS(160)
    
    X_all = np.array(X_all, dtype=np.float64)
    y_all = np.array(y_all, dtype=np.float64)
    
    X_all[np.isnan(X_all)] = 0
    
    # 调用模型得到预测值
    y_predict_all = model.predict(X_all)
    
    # 填充表4答案表格中对应的残差列
    table4_answer = pd.read_excel(os.path.join(root_path, table4_name))
    
    # print(table4_answer)
    
    # 使用 iloc 给特定列赋值
    table4_answer.iloc[2: 162, 9] = y_predict_all
    table4_answer.to_excel(os.path.join(root_path, table4_name), index=False, float_format='%.3f')
    
    print("success")
    

if __name__ == "__main__":
    predict_mRS()





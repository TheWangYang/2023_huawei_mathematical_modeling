import pandas as pd
import os
import numpy as np
from process_models.q1.q1a_check_patient import up_relative33
import matplotlib.pyplot as plt

# 设置数据的根路径
root_path = '/data1/wyy/projects/_competition/math_model_v2/data'

# 定义表1变量
excel_name1 = '1.xlsx'

# 定义表2变量
excel_name2 = '2.xlsx'

# 定义附表变量
excel_name3 = 'additional.xlsx'

# 读取原始Excel文件
table1 = pd.read_excel(os.path.join(root_path, excel_name1))
table2 = pd.read_excel(os.path.join(root_path, excel_name2))
table3 = pd.read_excel(os.path.join(root_path, excel_name3))

# 定义保存160个病人的数据的字典
patients_info = []

# 定义保存当前第i个病人信息的字典
info_dict = {}

# 遍历160个病人的数据
for i in range(160):
    # 创建保存病人信息的字典
    info_dict = {}
    
    # 得到三个表格中的每个病人第i行数据
    row1 = table1.loc[i]
    row2 = table2.loc[i]
    row3 = table3.loc[i]
    
    # 首次发病到首次检查时间点
    first_check_timedelta = row1[14]
    print("first_check_timedelta: {}".format(first_check_timedelta))
    
    info_dict['name'] = row2[0]
    
    # 检查号缓存列表
    info_dict['check_seq_buf'] = [row2[1]]
    # 得到首次检查对应的HM体积
    info_dict['HM_volum'] = [row2[2]]
    
    # 得到首次检查对应的时间点
    first_check_time = row3[2]
    
    # 得到首次检查时间点
    info_dict['check_time_buf'] = [first_check_timedelta]

    if not (row2[1] == row3[3] and row1[3]==row2[1]):
        print(f"id not equal, {row2[1]}, {row1[3]}")
    
    # 定义
    j=1
    
    print(i, first_check_timedelta)
    
    # 内层循环实现将病人的所有随访时间点距离首次发病的时间间隔等存储到info_dict中
    while 1+j*23 <len(row2):
        # 判断该病人是否存在随访点i + j * 23
        if not np.isnan(row2[1+j*23]):
            # 得到随访1的流水号
            info_dict['check_seq_buf'].append(int(row2[1+j*23]))
            
            # 得到随访1对应的HM体积
            info_dict['HM_volum'].append(row2[2+j*23])
            
            # 将随访1的时间点-首次检查时间点，换成小时，然后再加上发病到随访1的时间点
            info_dict['check_time_buf'].append((row3[2+2*j]-first_check_time).total_seconds()/3600 + first_check_timedelta)
            
            # 得到后续的随访结点，均按照上述进行处理
            j+=1
        else:
            break
    
    # info_dict['check_time_buf'] = np.array(info_dict['check_time_buf'])
    # 将其中的HM对应的数据转换为numpy数组的形式
    info_dict['HM_volum'] = np.array(info_dict['HM_volum'])
    # 将i编号的病人的信息增加到patients_info字典中
    patients_info.append(info_dict)

# 保存有血肿扩张风险的病人的列表
patient_sign = []
time_buf = []

# 枚举160个病人的信息
for i, info in enumerate(patients_info):
    # 定义
    has_patient = 0
    
    # 找到HM体积相对一首次检查增加超过6000的第一个结果存储到seq_buf中
    seq_buf = np.where((info['HM_volum']-info['HM_volum'][0])>6000)[0]
    
    # 找到HM体积相对一首次检查相对体积增加33%的索引值进行求与（因为是或）
    seq_buf = np.concatenate((seq_buf, up_relative33(info['HM_volum'])[0]))
    
    print("seq_buf: {}".format(seq_buf))
    
    # input()
    
    # 遍历seq_buf数组
    for seq in seq_buf:
        # print(info['check_time_buf'], seq-1)
        if info['check_time_buf'][seq] <= 48:
            has_patient = 1
            # print()
            # print(i+1, seq, info['HM_volum'][seq], info['HM_volum'][0], info['HM_volum'][seq] - info['HM_volum'][0], (info['HM_volum'][seq] - info['HM_volum'][0])/info['HM_volum'][0])
            # print(1)
            break
        
    # 已经小于48小时
    if has_patient:
        # 添加时间
        time_buf.append(info['check_time_buf'][seq])
        print("{:.2f}".format(info['check_time_buf'][seq]))
    else:
        time_buf.append(0)
        print(0)

# 绘图，横轴为病人编号，纵轴为发病时间
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure()

plt.plot(time_buf[:100])

# 添加标题和坐标轴标签
plt.title("发病时间")
plt.xlabel("病人编号(0-100)")
plt.ylabel("发病时间(h)")

# plt.show()
plt.savefig("/data1/wyy/projects/_competition/math_model_v2/data_process/output/time_of_patient.jpg")
    # print(has_patient)
    # patient_sign.append(has_patient)
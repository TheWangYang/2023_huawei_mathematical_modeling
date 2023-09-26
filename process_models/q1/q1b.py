import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from process_models.q1.q1b_dataloader import get_train_valid_q1

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import numpy as np
import shap

# 生成示例数据
X_train, X_test, y_train, y_test = get_train_valid_q1()

sm = SMOTE(random_state=0)

X_train, y_train = sm.fit_resample(X_train, y_train)
# X = np.random.random((100, 15))
# Y = np.random.randint(0, 2, (100,))

print(X_train.shape)


# X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)
# 转换数据为PyTorch的Tensor类型
X_train = torch.Tensor(X_train)
print(X_train.shape)
y_train = torch.LongTensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.LongTensor(y_test)

# 创建数据集和数据加载器

X_train = torch.cat([X_train, X_test], dim=0)
y_train = torch.cat([y_train, y_test], dim=0)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

# 构建神经网络模型
input_n = X_train.shape[-1]
start_n = 64
middle_n = [64, 64, 32, 16, 2]

print(input_n)
torch.manual_seed(42)

# 构建简单的
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_n, start_n),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        
        for i, num in enumerate(middle_n):
            if i==0:
                self.fc.append(nn.Sequential(nn.Linear(start_n, num),
                                             nn.ReLU()))
            else:
                self.fc.append(nn.Sequential(nn.Linear(middle_n[i-1], num),
                                             nn.ReLU()))

        self.fc.append(nn.Softmax(dim=1))

    def forward(self, x):
        x = self.fc(x)
        return x

model = Net()
model.train()
if True:
    # model.load_state_dict(torch.load('model_weights_base_info.pth'))
    # model.load_state_dict(torch.load('model_weights.pth'))
    print('loaded')

# 加载模型参数
# model.load_state_dict(torch.load('your_model.pth'))

# 初始化SHAP解释器
explainer = shap.DeepExplainer(model, X_train)

# 计算特征重要性
# shap_values = explainer(X_test_tensor)

shap_values = explainer.shap_values(X_test)

print(len(shap_values), X_test.shape, shap_values[1].shape)

# torch.mean(shap_values[0], dim=0)
# 定义柱状图的数据
categories = [f'feature{i}' for i in range(shap_values[0].shape[1])]
values = np.mean(shap_values[0], axis=0)
values2 = np.mean(shap_values[1], axis=0)
# 创建一个图形窗口
plt.figure()

# 绘制横向柱状图
plt.barh(categories[:values.shape[0]], values)
# plt.barh(categories[:values.shape[0]], values)
# 添加标题和坐标轴标签
# plt.title("横向柱状图")
# plt.xlabel("数值")
# plt.ylabel("类别")

# 显示图形
plt.show()

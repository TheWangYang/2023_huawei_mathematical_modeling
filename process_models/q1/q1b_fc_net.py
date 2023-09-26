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

# X_train = torch.cat([X_train, X_test], dim=0)
# y_train = torch.cat([y_train, y_test], dim=0)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

# 构建神经网络模型
input_n = X_train.shape[-1]
start_n = 64
middle_n = [64, 64, 32, 16, 2]

print(input_n)
torch.manual_seed(42)


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
    model.load_state_dict(torch.load('model_weights.pth'))
    print('loaded')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
# 训练模型
num_epochs = 1000
with torch.no_grad():
    outputs = model(X_test)
    # print(outputs.data)
    _, predicted = torch.max(outputs.data, 1)

    accuracy = (predicted == y_test).sum().item() / len(y_test)

    print("Accuracy:", accuracy)

for epoch in range(num_epochs):
    # scheduler.step()
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.data)
        # print(outputs.data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(loss)

    if epoch%10 ==1 :
        with torch.no_grad():
            outputs = model(X_train)
            print(loss)
            _, predicted = torch.max(outputs.data, 1)

            print(outputs.data[-1])
            print((y_train==1).sum()/len(y_train))
            # 输出准确率
            accuracy = (predicted == y_train[:]).sum().item() / len(y_train)
            print("TP:", y_train[predicted == y_train].sum().item()/(y_train==1).sum())
            print("TF:", ((predicted == y_train).sum().item()-y_train[predicted == y_train].sum().item())/(y_train==0).sum())
            
            print("Accuracy:", accuracy)
            # print("Accuracy:", accuracy)

    
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 在测试集上进行预测
with torch.no_grad():
    outputs = model(X_test)
    
    _, predicted = torch.max(outputs.data, 1)
    print(outputs.data)

    # 输出准确率
    accuracy = (predicted == y_test).sum().item() / len(y_test)

    print("TP:", y_test[predicted == y_test].sum().item()/(y_test==1).sum())
    print("TF:", ((predicted == y_test).sum().item()-y_test[predicted == y_test].sum().item())/(y_test==0).sum())
    
    print("Accuracy:", accuracy)
    print("y_test:", y_test)

# Importance analythis

torch.save(model.state_dict(), 'model_weights.pth')
    # print('X_text', X_test)

# # Importance analythis
# explainer = shap.Explainer(model, X_train)

# shap_values = explainer(X_test)

# # 绘制特征重要性图
# shap.summary_plot(shap_values, X_test_tensor, plot_type='bar')

# # 显示图形
# plt.show()
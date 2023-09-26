import torch
import torch.nn as nn
import torch.optim as optim


# 构建SVM分类器
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
    
# 假设有训练数据和标签
train_data = torch.randn(100, 100)
labels = torch.randint(0, 2, (100,))


svm_model = SVM(100)
optimizer = optim.SGD(svm_model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    # 前向传播
    outputs = svm_model(train_data)

    # 计算损失
    loss = torch.mean(torch.clamp(1 - labels * outputs, min=0))

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练过程中的损失
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        


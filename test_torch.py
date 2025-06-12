import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型结构
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 初始化模型
input_size = 32  # 输入节点数量
hidden_size1 = 64  # 隐藏层1节点数
hidden_size2 = 128  # 隐藏层2节点数
output_size = 10  # 输出节点数

model = SimpleNeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 数据预处理
train_data = (torch.randn(32),)  # 输入数据
train_labels = torch.randint(0, 9, (32,))  # 输出标签

# 进行训练循环
for epoch in range(10):
    for i in range(len(train_data)):
        x = train_data[i]
        y = train_labels[i]
        # 前向传播
        outputs = model(x)
        # 计算损失
        loss = loss_fn(outputs, y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试模型
test_data = (torch.randn(32),)
test_labels = torch.randint(0, 9, (32,))
print("测试结果:", test_labels)
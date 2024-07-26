import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 配置参数
config = {
    'batch_size': 128,
    'num_epochs': 30,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'num_classes': 10,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# 数据增强和归一化
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪和填充
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = F.relu(out)
        return out


# 定义残差网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)


# 初始化模型
net = ResNet18(num_classes=config['num_classes'])
net.to(config['device'])

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])

# 定义学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 记录损失值、总体准确率和类别准确率
train_losses = []
test_losses = []
overall_accuracies = []
class_correct = np.zeros(config['num_classes'])
class_total = np.zeros(config['num_classes'])


# 训练函数
def train(epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(config['device']), labels.to(config['device'])

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.3f},', end=" ")
    train_losses.append(epoch_loss)


# 测试函数
def test(epoch):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if epoch == config['num_epochs'] - 1:
                for i in range(labels.size(0)):
                    label = labels[i].item()  # 转为整数类型
                    class_total[label] += 1
                    class_correct[label] += (predicted[i] == label).item()

    test_losses.append(running_loss / len(test_loader))

    accuracy = 100 * correct / total
    overall_accuracies.append(accuracy)

    if epoch == config['num_epochs'] - 1:
        print(f'Accuracy: {accuracy:.2f}%')
        for i in range(config['num_classes']):
            if class_total[i] > 0:
                class_accuracy = 100 * class_correct[i] / class_total[i]
            else:
                class_accuracy = 0.0
            print(f'Class {i} Accuracy: {class_accuracy:.2f}%')
    else:
        print(f'Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    for epoch in range(config['num_epochs']):
        train(epoch)
        test(epoch)

    epochs = list(range(1, config['num_epochs'] + 1))

    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Testing Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid()
    plt.show()


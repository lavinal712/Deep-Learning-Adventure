import os
import torch
from torch import nn
from utils.load_kaggle_cat_dog import *


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_channels, use_1x1conv=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        identity = x
        output = self.relu(self.bn1(self.conv1(x)))
        if self.conv3 is not None:
            identity = self.conv3(x)
        output = self.relu(self.bn2(self.conv2(output)) + identity)
        return output


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(in_channels=64, num_channels=64, blocks=2, stride=1)
        self.layer2 = self._make_layer(in_channels=64, num_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(in_channels=128, num_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(in_channels=256, num_channels=512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, num_channels, blocks, stride=1):
        layer = []
        layer.append(ResidualBlock(in_channels, num_channels, use_1x1conv=True, stride=1))
        for _ in range(1, blocks):
            layer.append(ResidualBlock(num_channels, num_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        output = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        output = self.layer4(self.layer3(self.layer2(self.layer1(output))))
        output = self.fc(torch.flatten(self.avgpool(output), 1))
        return output


def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def load_model(device):
    if not os.path.exists('./checkpoints'):
        return ResNet(num_classes=2).to(device), 0

    model_list = sorted(os.listdir('./checkpoints'), key=lambda x: int(x.split('_')[-1][: -3]))
    last_model_name = model_list[-1]
    trained_epoch = int(last_model_name.split('_')[-1][: -3])
    model = ResNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join('./checkpoints', last_model_name)))
    print(f'Model {last_model_name} Loaded')

    return model, trained_epoch


def save_model(model, epoch):
    model_name = f'ResNet_{epoch + 1}.pt'
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('./checkpoints', model_name))
    print(f'Model {model_name} Saved')


def main():
    train_loader, val_loader, test_loader = load_kaggle_cat_dog(batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, trained_epoch = load_model(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    num_steps = len(train_loader)
    for epoch in range(trained_epoch, num_epochs):
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Step {step + 1}/{num_steps}, Loss: {l.item()}")

        val_acc = eval_acc(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc}")

        save_model(model, epoch)

    test_acc = eval_acc(model, test_loader, device)
    print(f"Test Accuracy: {test_acc}")


if __name__ == '__main__':
    main()

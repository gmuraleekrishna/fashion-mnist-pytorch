import torch.nn as nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # torch.Size([1, 32, 24, 24])
        self.relu1 = nn.ReLU()  # torch.Size([1, 32, 24, 24])
        self.batch_norm1 =  nn.BatchNorm2d(num_features=32)  # torch.Size([1, 32, 24, 24])
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=(2, 2))  # torch.Size([1, 64, 12, 12])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, stride=1)  # torch.Size([1, 64, 12, 12])
        self.relu2 = nn.ReLU()  # torch.Size([1, 64, 12, 12])
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)  # torch.Size([1, 64, 12, 12])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # torch.Size([1, 64, 6, 6])
        self.do1 = nn.Dropout()
        self.fc1 =  nn.Linear(64*6*6, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(self.batch_norm1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool2(self.batch_norm2(x))
        x = self.do1(x)
        x = x.view(-1, 64*6*6)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 5×5 Convolutional Layer with 32 filters, stride 1 and padding 1.
# • ReLU Activation Layer.
# • Batch Normalization Layer
# • 2×2 Max Pooling Layer with a stride of 2.
# • 3×3 Convolutional Layer with 64 filters, stride 1 and padding 1.
# • ReLU Activation Layer.
# • Batch Normalization Layer.
# • 2×2 Max Pooling Layer with a stride of 2.
# • Fully-connected layer with 1024 output units.
# • ReLU Activation Layer.
# • Fully-connected layer with 10 output units.
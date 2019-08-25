from torch import nn

# --summary --cuda --drops 0.25 0.25 0.5 --epoch 15 --batch 50 Test accuracy: 89.20% Test Loss: 0.30


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)  # torch.Size([1, 32, 24, 24])
		self.relu1 = nn.ReLU()  # torch.Size([1, 32, 24, 24])
		self.batch_norm1 = nn.BatchNorm2d(num_features=32)  # torch.Size([1, 32, 24, 24])
		self.pool1 = nn.MaxPool2d(stride=2, kernel_size=(2, 2))  # torch.Size([1, 64, 12, 12])
		# self.drop1 = nn.Dropout(p=0.25)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, stride=1)  # torch.Size([1, 64, 12, 12])
		self.relu2 = nn.ReLU()  # torch.Size([1, 64, 12, 12])
		self.batch_norm2 = nn.BatchNorm2d(num_features=64)  # torch.Size([1, 64, 12, 12])
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # torch.Size([1, 64, 6, 6])
		# self.drop2 = nn.Dropout(p=0.25)

		self.fc1 = nn.Linear(64*6*6, 1024)
		self.relu3 = nn.ReLU()
		# self.drop3 = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(in_features=1024, out_features=10)

	def forward(self, x):
		x = self.relu1(self.conv1(x))
		x = self.pool1(self.batch_norm1(x))
		# x = self.drop1(x)
		x = self.relu2(self.conv2(x))
		x = self.pool2(self.batch_norm2(x))
		# x = self.drop2(x)
		x = x.view(-1, 64*6*6)  # Flatten
		x = self.relu3(self.fc1(x))
		# x = self.drop3(x)
		x = self.fc2(x)
		return x

from torch import nn

# --summary --cuda --drops 0.25 0.25 0.5 --epoch 15 --batch 50 Test accuracy: 93.14%%
# Test Loss: 0.21


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.relu1 = nn.ReLU()  # [-1, 32, 28, 28]
		self.batch_norm1 = nn.BatchNorm2d(num_features=32)  # [-1, 32, 28, 28]
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.relu2 = nn.ReLU()  # [-1, 64, 28, 28]
		self.batch_norm2 = nn.BatchNorm2d(num_features=64) # [-1, 64, 14, 14]
		self.pool1 = nn.MaxPool2d(stride=2, kernel_size=(2, 2))
		self.drop1 = nn.Dropout(p=0.25)

		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1)
		self.relu3 = nn.ReLU()  # [-1, 64, 12, 12]
		self.batch_norm3 = nn.BatchNorm2d(num_features=64)
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
		self.relu4 = nn.ReLU()  # [-1, 128, 10, 10]
		self.batch_norm4 = nn.BatchNorm2d(num_features=128)
		self.pool2 = nn.MaxPool2d(stride=2, kernel_size=(2, 2))  # [-1, 128, 5, 5]
		self.drop2 = nn.Dropout(p=0.25)

		self.fc1 = nn.Linear(128*5*5, 1024)  # [-1, 1024]
		self.relu5 = nn.ReLU()
		self.batch_norm5 = nn.BatchNorm1d(num_features=1024)
		self.drop3 = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(in_features=1024, out_features=10)  # [-1, 10]

	def forward(self, x):
		x = self.batch_norm1(self.relu1(self.conv1(x)))
		x = self.batch_norm2(self.relu2(self.conv2(x)))
		x = self.pool1(x)
		x = self.drop1(x)

		x = self.batch_norm3(self.relu3(self.conv3(x)))
		x = self.batch_norm4(self.relu4(self.conv4(x)))
		x = self.pool2(x)
		x = self.drop2(x)

		x = x.view(-1, 128*5*5)  # Flatten
		x = self.batch_norm5(self.relu5(self.fc1(x)))
		x = self.drop3(x)
		x = self.fc2(x)
		return x

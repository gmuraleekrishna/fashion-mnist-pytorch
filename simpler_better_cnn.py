from torch import nn

# --summary --cuda --drops 0.25 0.25 0.5 --epoch 15 --batch 50 Test accuracy: 93.05%
# Test Loss: 0.21
#
#
# Test accuracy: 94.07%
# Test Loss: 0.17


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=1)
		self.relu1_1 = nn.ReLU()  # [-1, 32, 28, 28]
		self.batch_norm1_1 = nn.BatchNorm2d(num_features=64)  # [-1, 32, 28, 28]
		self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=1)
		self.relu1_2 = nn.ReLU()  # [-1, 64, 28, 28]
		self.batch_norm1_2 = nn.BatchNorm2d(num_features=32) # [-1, 64, 14, 14]`
		self.pool1_1 = nn.MaxPool2d(stride=2, kernel_size=2)
		# self.drop1_1 = nn.Dropout(p=0.4)
		#
		# self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
		# self.relu2_1 = nn.ReLU()  # [-1, 64, 12, 12]
		# self.batch_norm2_1 = nn.BatchNorm2d(num_features=128)
		# self.pool2_1 = nn.MaxPool2d(stride=2, kernel_size=2)  # [-1, 128, 5, 5]
		# self.drop2_1 = nn.Dropout(p=0.4)

		self.fc3_1 = nn.Linear(32*12*12, 1024)  # [-1, 1024]
		self.relu3_1 = nn.ReLU()
		self.drop3_1 = nn.Dropout(p=0.5)
		self.fc3_2 = nn.Linear(in_features=1024, out_features=10)  # [-1, 10]

	def forward(self, x):
		x = self.batch_norm1_1(self.relu1_1(self.conv1_1(x)))
		x =self.batch_norm1_2( self.relu1_2(self.conv1_2(x)))
		# x = self.batch_norm1_3(self.relu1_3(self.conv1_3(x)))
		x = self.pool1_1(x)
		# x = self.drop1_1(x)

		# x = self.batch_norm2_1(self.relu2_1(self.conv2_1(x)))
		# # x = self.batch_norm2_2(self.relu2_2(self.conv2_2(x)))
		# x = self.pool2_1(x)
		# # x = self.drop2_1(x)

		x = x.view(-1, 32*12*12)  # Flatten
		x = self.relu3_1(self.fc3_1(x))
		x = self.drop3_1(x)
		x = self.fc3_2(x)
		return x

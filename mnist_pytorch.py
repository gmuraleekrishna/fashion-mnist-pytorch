import torch
import torchvision
from torch import nn, optim  # Contains several Pytorch optimizer classes
from torchvision import transforms
from torch.utils.data import *

import cnn


transform = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=4, pad_if_needed=False, fill=0, padding_mode='constant'),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ]
)

# train_idx, valid_idx, testidx = list(range(num_training)), list(range(num_validation)), list(range(num_test))
#
# train_loader = DataLoader(mnist_dataset, batch_size=50,sampler=SubsetRandomSampler(train_idx), num_workers=0)
# val_loader = DataLoader(mnist_dataset, batch_size=50,sampler=SubsetRandomSampler(valid_idx), num_workers=0)
# test_loader = DataLoader(mnist_dataset, batch_size=50,sampler=SubsetRandomSampler(valid_idx), num_workers=0)

train_set = torchvision.datasets.FashionMNIST(root='./FashionMNIST', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

val_set = torchvision.datasets.FashionMNIST(root='./FashionMNIST', train=False, download=False, transform=transform)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)

net = cnn.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
# output = net(torch.rand(1, 1, 28, 28))
# print(output.shape)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
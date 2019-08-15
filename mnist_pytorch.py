import torch
import torchvision
from torch import nn, optim  # Contains several Pytorch optimizer classes
from torchvision import transforms
from torch.utils.data import *

import cnn

NUM_EPOCH = 5
BATCH_SIZE = 100

transform = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=4, pad_if_needed=False, fill=0, padding_mode='constant'),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ]
)

train_set = torchvision.datasets.FashionMNIST(root='./', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_set = torchvision.datasets.FashionMNIST(root='./', train=False, download=False, transform=transform)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

net = cnn.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(NUM_EPOCH):
    for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, NUM_EPOCH, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

            if i % 600 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'mnnist{:d}_{:d}.pth'.format(epoch, i))

torch.save({
    'model_state_dict': net.state_dict()
})
import torch
from torch import nn, optim  # Contains several Pytorch optimizer classes
from torch.autograd import Variable

from data_loader import load_data
import cnn

NUM_EPOCH = 5
BATCH_SIZE = 100

net = cnn.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

train_loader, val_loader = load_data(BATCH_SIZE)

total_step = len(train_loader)
loss_list = []
acc_list = []

net.cuda()

for epoch in range(NUM_EPOCH):
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)
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

        if (batch_idx + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, NUM_EPOCH, batch_idx + 1, total_step, loss.item(),
                          (correct / total) * 100))

        if batch_idx % 600 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'mnnist{:d}_{:d}.pth'.format(epoch, batch_idx))


# testing
correct_cnt, ave_loss = 0, 0
total_cnt = 0
for epoch in range(NUM_EPOCH):
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images, volatile=True), Variable(labels, volatile=True)
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        _, pred_label = torch.max(outputs.data, 1)
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
            print
            '==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx + 1, ave_loss, correct * 1.0 / total)
torch.save({
    'model_state_dict': net.state_dict()
})


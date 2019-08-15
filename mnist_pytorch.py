import torch
from torch import nn, optim  # Contains several Pytorch optimizer classes

from data_loader import load_data
import cnn

NUM_EPOCH = 5
BATCH_SIZE = 100

net = cnn.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

train_loader, val_loader = load_data()

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

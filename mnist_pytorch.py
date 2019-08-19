import torch
from torch import nn, optim  # Contains several Pytorch optimizer classes
from torch.autograd import Variable
import argparse
import numpy as np

from data_loader import load_data
import cnn

NUM_EPOCH = 30
BATCH_SIZE = 500


def forward_pass(net, images, labels, optimizer, criterion, train=False):
    outputs = net(images)
    loss = criterion(outputs, labels)
    # Backprop and perform Adam optimisation
    optimizer.zero_grad()
    if train:
        loss.backward()
        optimizer.step()
    total = labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return loss, correct, total


def network_pass(net, batch_idx, images, labels, optimizer, criterion, train=False):
    if args.cuda:
        images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images, volatile=True), Variable(labels, volatile=True)
    loss, correct, total = forward_pass(net, images, labels, optimizer, criterion, train=train)
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch + 1, NUM_EPOCH, batch_idx + 1, total, loss.item(),
                  (correct / total) * 100))
    return loss, correct, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST on pytorch')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='use cuda')
    parser.add_argument('--val', dest='validate_only', help='validate model')
    parser.add_argument('--file', dest='validation_file', help='validate model')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    args = parser.parse_args()

    if args.validate_only and (args.validation_file is None):
        parser.error("--val requires --validation_file")

    net = cnn.CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    lowest_loss = np.Inf
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    train_loader, val_loader = load_data(BATCH_SIZE)

    if args.cuda:
        net.to(device)

    print('Size of training set: ', len(train_loader))
    print('Size of val set: ', len(val_loader))

    if not args.validate_only:
        for epoch in range(NUM_EPOCH):
            avg_acc = 0
            avg_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                if args.cuda:
                    images, labels = images.to(device), labels.to(device)
                loss, correct, total = network_pass(net, batch_idx, images, labels, optimizer, criterion, train=True)
                avg_acc += (correct / total) * 100
                avg_loss += loss.item()
            if args.tensorboard:
                writer.add_scalar('Accuracy/train',  avg_acc / len(train_loader), epoch)
                writer.add_scalar('Loss/train', avg_loss / len(train_loader), epoch)
                if avg_loss < lowest_loss:
                    lowest_loss = avg_loss
                    torch.save({
                        'model_state_dict': net.state_dict()
                    }, 'fashion-mnist.pth')

    if args.validate_only:
        net.load_state_dict(args.validation_file)
        net.eval()
        print('Network loaded')
    for epoch in range(NUM_EPOCH):
        avg_acc = 0
        avg_loss = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            loss, correct, total = network_pass(net, batch_idx, images, labels, optimizer, criterion, train=False)
            avg_acc += (correct / total) * 100
            avg_loss = avg_loss * 0.9 + loss.item() * 0.1
        if args.tensorboard:
            writer.add_scalar('Accuracy/Val', avg_acc / len(val_loader), epoch)
            writer.add_scalar('Loss/Val', avg_loss, epoch)

if args.tensorboard:
    writer.close()

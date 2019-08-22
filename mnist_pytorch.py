import torch
from torch import nn, optim  # Contains several Pytorch optimizer classes
from torch.autograd import Variable
import argparse
import numpy as np
from torchsummary import summary

from data_loader import load_data
import cnn

NUM_EPOCH = 5
BATCH_SIZE = 50


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


def network_pass(net, batch_idx, images, labels, optimizer, criterion, train=False, verbose=False):
	if args.cuda:
		images, labels = images.cuda(), labels.cuda()
		images, labels = Variable(images), Variable(labels)
	loss, correct, total = forward_pass(net, images, labels, optimizer, criterion, train=train)
	if train and batch_idx % 50 == 0 and verbose:
		print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
		      .format(epoch + 1, NUM_EPOCH, batch_idx + 1, total, loss.item(), (correct / total) * 100))
	elif not train and verbose:
		print('Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(loss.item(), (correct / total) * 100))
	return loss, correct, total


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FashionMNIST on pytorch')
	parser.add_argument('--cuda', dest='cuda', action='store_true', help='use cuda')
	parser.add_argument('--test', dest='test_only', action='store_true', help='test model')
	parser.add_argument('--file', dest='test_file', help='test model file')
	parser.add_argument('--summary', dest='summary', action='store_true', help='show network summary')
	parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
	args = parser.parse_args()

	if args.test_only and (args.test_file is None):
		parser.error("--test requires --file")

	net = cnn.CNN()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
	device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
	lowest_loss = np.Inf
	if args.tensorboard:
		from torch.utils.tensorboard import SummaryWriter

		writer = SummaryWriter()

	train_loader, val_loader, test_loader = load_data(BATCH_SIZE)

	net.to(device)
	if args.summary:
		summary(net, (1, 28, 28))

	print('Size of training set: ', len(train_loader))
	print('Size of val set: ', len(val_loader))

	if not args.test_only:
		for epoch in range(NUM_EPOCH):
			avg_acc = 0
			avg_loss = 0
			avg_val_acc = 0
			avg_val_loss = 0
			for batch_idx, (images, labels) in enumerate(train_loader):
				if args.cuda:
					images, labels = Variable(images.cuda()), Variable(labels.cuda())
				loss, correct, total = network_pass(net, batch_idx, images, labels, optimizer, criterion, train=True,
				                                    verbose=True)
				avg_acc += (correct / total) * 100
				avg_loss += loss.item()
				if args.tensorboard:
					writer.add_scalar('Accuracy/train', avg_acc / len(train_loader), epoch)
					writer.add_scalar('Loss/train', avg_loss / len(train_loader), epoch)
			for batch_idx, (images, labels) in enumerate(val_loader):
				val_loss, val_correct, val_total = network_pass(net, batch_idx, images, labels, optimizer, criterion,
				                                                train=False, verbose=True)
				avg_val_acc += (val_correct / val_total) * 100
				avg_val_loss = avg_val_loss * 0.9 + val_loss.item() * 0.1
				if args.tensorboard:
					writer.add_scalar('Accuracy/Val', avg_val_acc / len(val_loader), epoch)
					writer.add_scalar('Loss/Val', avg_val_loss, epoch)
			if avg_val_loss < lowest_loss:
				lowest_loss = avg_val_loss
				torch.save({
					'model_state_dict': net.state_dict()
				}, 'fashion-mnist.pth')

	if args.test_only:
		net.load_state_dict(torch.load(args.test_file)['model_state_dict'])
		net.eval()
		print('Network loaded')
	avg_acc = 0.0
	avg_loss = 0.0
	print()
	for batch_idx, (images, labels) in enumerate(test_loader):
		test_loss, test_correct, test_total = network_pass(net, batch_idx, images, labels, optimizer, criterion,
		                                                   train=False, verbose=False)
		avg_acc += test_correct
		avg_loss += test_loss.item()
	print('Test accuracy: {:.2f}\nTest Loss: {:.2f}'.format(avg_acc/len(test_loader) * 100, avg_loss/len(test_loader)))

if args.tensorboard:
	writer.close()

import torch
from torch import nn, optim  # Contains several Pytorch optimizer classes
import argparse
import numpy as np
from torchsummary import summary
from torch.autograd import Variable
import os

from data_loader import load_data
import cnn
from network_helper import network_pass, print_status

NUM_EPOCH = 5
BATCH_SIZE = 50

os.environ['OMP_NUM_THREADS'] = '1'

def train(net, train_loader, device, tensorboard=False,  writer=None):
	avg_acc = 0
	avg_loss = 0
	net = net.train()
	for batch_id, (train_images, train_labels) in enumerate(train_loader):
		train_images, train_labels = train_images.to(device), train_labels.to(device).long()
		adam_optimizer.zero_grad()
		train_loss, train_correct, train_total = network_pass(net, train_images, train_labels,
		                                                      adam_optimizer, cross_entropy_loss, train=True)
		if batch_id % 50 == 0:
			print_status(batch_idx=batch_id, correct=train_correct, total=train_total,
			             num_epochs=args.num_epochs, epoch=epoch, train=True, loss=train_loss)
		avg_acc += (train_correct / train_total) * 100
		avg_loss += train_loss.item()
		if tensorboard:
			writer.add_scalar('Accuracy/train', avg_acc / len(train_loader), epoch)
			writer.add_scalar('Loss/train', avg_loss / len(train_loader), epoch)
	return avg_acc, avg_loss


def eval(net, val_loader, device, tensorboard=False, writer=None):
	avg_acc = 0
	avg_loss = 0
	net = net.eval()
	with torch.no_grad():
		for batch_id, (val_images, val_labels) in enumerate(val_loader):
			val_images, val_labels = val_images.to(device), val_labels.to(device).long()
			val_loss, val_correct, val_total = network_pass(net, val_images, val_labels, adam_optimizer,
			                                                cross_entropy_loss, train=False)
			print_status(batch_idx=batch_id, num_epochs=args.num_epochs,
			             total=val_total, correct=val_correct, train=False, epoch=epoch, loss=val_loss)
			avg_acc += (val_correct / val_total) * 100
			avg_loss = avg_val_loss * 0.9 + val_loss.item() * 0.1
			if tensorboard:
				writer.add_scalar('Accuracy/Val', avg_acc / len(val_loader), epoch)
				writer.add_scalar('Loss/Val', avg_loss, epoch)
	return avg_acc, avg_loss


def test(net, device, test_loader):
	net.eval()
	torch.no_grad()
	avg_acc = 0
	avg_loss = 0
	for batch_id, (test_images, test_labels) in enumerate(test_loader):
		test_images, test_labels = test_images.to(device), test_labels.to(device).long()
		test_loss, test_correct, test_total = network_pass(net, test_images, test_labels, adam_optimizer,
		                                                   cross_entropy_loss, train=False)
		avg_acc += test_correct
		avg_loss += test_loss.item()
	print('====================================================')
	print('Test accuracy: {:.2f}%\nTest Loss: {:.2f}'.format(avg_acc / len(test_loader) * 100,
	                                                        avg_loss / len(test_loader)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FashionMNIST on pytorch')
	parser.add_argument('--cuda', dest='cuda', action='store_true', help='use CUDA')
	parser.add_argument('--test', dest='test_only', action='store_true', help='test model')
	parser.add_argument('--file', dest='test_file', help='test model file')
	parser.add_argument('--summary', dest='summary', action='store_true', help='show network summary')
	parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
	parser.add_argument('--epoch', dest='num_epochs', help='number of epochs', type=int, default=NUM_EPOCH)
	parser.add_argument('--batch', dest='batch_size', type=int, help='batch size', default=BATCH_SIZE)
	args = parser.parse_args()

	print(args.batch_size)
	if args.test_only and (args.test_file is None):
		parser.error("--test requires --file")

	net = cnn.CNN()
	cross_entropy_loss = nn.CrossEntropyLoss()
	adam_optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
	device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
	lowest_loss = np.Inf
	writer = None
	if args.tensorboard:
		from torch.utils.tensorboard import SummaryWriter

		writer = SummaryWriter()

	train_loader, val_loader, test_loader = load_data(args.batch_size)

	net.to(device)
	if args.summary:
		summary(net, (1, 28, 28))

	print('Size of training set: ', len(train_loader))
	print('Size of val set: ', len(val_loader))

	if not args.test_only:
		for epoch in range(NUM_EPOCH):
			avg_train_acc, avg_train_loss = train(net=net, device=device, train_loader=train_loader, writer=writer,
			                          tensorboard=args.tensorboard)
			avg_val_acc, avg_val_loss = eval(net=net, device=device,
			                                 val_loader=val_loader, writer=writer, tensorboard=args.tensorboard)
			if avg_val_loss < lowest_loss:
				lowest_loss = avg_val_loss
				torch.save({
					'model_state_dict': net.state_dict()
				}, 'fashion-mnist.pth')

	if args.test_only:
		net.load_state_dict(torch.load(args.test_file)['model_state_dict'])
		print('Network loaded')
	test(net=net, device=device, test_loader=test_loader)

	if args.tensorboard:
		writer.close()

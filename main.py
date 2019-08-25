import torch
from torch import nn, optim  # Contains several Pytorch optimizer classes
import argparse
import numpy as np
from torchsummary import summary
import os
from summarywriter import SummaryWriter

from data_loader import load_data
import cnn as cnn
from network_helper import train, evaluate, test

NUM_EPOCH = 5
BATCH_SIZE = 50


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FashionMNIST on pytorch')
	parser.add_argument('--cuda', dest='cuda', action='store_true', help='use CUDA', default=False)
	parser.add_argument('--test', dest='test_only', action='store_true', help='test model', default=False)
	parser.add_argument('--file', dest='test_file', help='test model file')
	parser.add_argument('--summary', dest='summary', action='store_true', help='show network summary', default=False)
	parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', default=False)
	parser.add_argument('--epoch', dest='num_epochs', help='number of epochs', type=int, default=NUM_EPOCH)
	parser.add_argument('--batch', dest='batch_size', type=int, help='batch size', default=BATCH_SIZE)
	parser.add_argument('--drops', dest='drop_probs', nargs=3, type=float,
	                    help='dropout probability', default=[0, 0, 0])
	parser.add_argument('--init_weights', dest='init_weights', action='store_true', help='init weights for conv layers',
	                    default=False)
	args = parser.parse_args()

	if args.test_only and (args.test_file is None):
		parser.error("--test requires --file")

	net = cnn.CNN()
	cross_entropy_loss = nn.CrossEntropyLoss()
	adam_optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
	device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
	lowest_loss = np.Inf
	writer = None
	if args.tensorboard:
		home = os.environ['HOME']
		writer = SummaryWriter(home + '/25Aug_cnn_new.json')

	train_loader, val_loader, test_loader = load_data(args.batch_size)

	net.to(device)
	if args.summary:
		summary(net, (1, 28, 28))
		print('Initialising weights: ', args.init_weights)
		print('Drop probabilities: ', args.drop_probs)
		print('Batch size: ',  args.batch_size)
		print('Epochs:', args.num_epochs)

	if not args.test_only:
		for epoch in range(args.num_epochs):
			avg_train_acc, avg_train_loss = train(net=net, device=device, optimizer=adam_optimizer,
			                                      train_loader=train_loader, writer=writer, epoch=epoch,
			                                      loss_criterion=cross_entropy_loss,
			                                      num_epochs=args.num_epochs, tensorboard=args.tensorboard)
			avg_val_acc, avg_val_loss = evaluate(net=net, device=device, val_loader=val_loader, writer=writer,
			                                     tensorboard=args.tensorboard, loss_criterion=cross_entropy_loss,
			                                     epoch=epoch)
			if avg_val_loss < lowest_loss:
				lowest_loss = avg_val_loss
				torch.save({
					'model_state_dict': net.state_dict()
				}, 'fashion-mnist.pth')

	if args.test_only:
		net.load_state_dict(torch.load(args.test_file)['model_state_dict'])
		print('Network loaded')
	test(net=net, device=device, test_loader=test_loader, loss_criterion=cross_entropy_loss)

	if args.tensorboard:
		writer.close()

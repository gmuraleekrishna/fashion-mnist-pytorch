import torch
from torch import nn, optim
import argparse
from torchsummary import summary
import os
from summarywriter import SummaryWriter
import numpy as np

from data_loader import load_data
import cnn as cnn
from network_helper import Trainer, Evaluator


NUM_EPOCH = 5
BATCH_SIZE = 50


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='FashionMNIST on pytorch')
	parser.add_argument('--test', dest='test_only', action='store_true', help='test model', default=False)
	parser.add_argument('--file', dest='test_file', help='test model file')
	parser.add_argument('--summary', dest='summary', action='store_true', help='show network summary', default=False)
	parser.add_argument('--epoch', dest='num_epochs', help='number of epochs', type=int, default=NUM_EPOCH)
	parser.add_argument('--batch', dest='batch_size', type=int, help='batch size', default=BATCH_SIZE)
	args = parser.parse_args()

	if args.test_only and (args.test_file is None):
		parser.error("--test requires --file")

	model = cnn.CNN()
	cross_entropy_loss = nn.CrossEntropyLoss()
	adam_optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	home = os.environ['HOME']
	writer = SummaryWriter(home + '/25Aug_cnn_new.json')
	lowest_loss = np.Inf
	train_loader, val_loader, test_loader = load_data(args.batch_size)

	model.to(device)
	
	if args.summary:
		summary(model, (1, 28, 28))
		print('Batch size: ',  args.batch_size)
		print('Epochs:', args.num_epochs)
	
	trainer = Trainer(model=model, device=device, optimizer=adam_optimizer, loader=train_loader, writer=writer,
			                                      loss_function=cross_entropy_loss)
	evaluator = Evaluator(model=model, device=device, loader=val_loader, writer=writer,
			                                      loss_function=cross_entropy_loss)
	if not args.test_only:
		for epoch in range(args.num_epochs):
			avg_train_acc, avg_train_loss = trainer.train(epoch=epoch)
			print('Epoch {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%'
			      .format(epoch + 1, avg_train_loss, avg_train_acc))
			avg_val_acc, avg_val_loss = evaluator.evaluate(epoch=epoch)
			print('Epoch {}, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
			      .format(epoch + 1, avg_val_loss, avg_val_acc))
			if avg_val_loss < lowest_loss:
				lowest_loss = avg_val_loss
				torch.save({
					'model_state_dict': model.state_dict()
				}, 'fashion-mnist.pth')

	if args.test_only:
		model.load_state_dict(torch.load(args.test_file)['model_state_dict'])
		print('Network loaded')
	tester = Evaluator(model=model, device=device, loader=test_loader, loss_function=cross_entropy_loss, is_test=True)
	avg_acc, avg_loss = tester.evaluate()
	print('====================================================')
	print('Test accuracy: {:.2f}%\nTest Loss: {:.2f}'.format(avg_acc, avg_loss))

	writer.close()

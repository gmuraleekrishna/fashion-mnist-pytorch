import torch
from torch import nn, optim
import argparse
from torchsummary import summary
import os
from summarywriter import SummaryWriter
import numpy as np
from ignite.handlers import ModelCheckpoint

from data_loader import load_data
import new_cnn as cnn
from network_helper import Evaluator

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from tqdm import tqdm

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
	adam_optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	home = os.environ['HOME']
	writer = SummaryWriter(home + '/log.json')
	lowest_loss = np.Inf
	train_loader, val_loader, test_loader = load_data(args.batch_size)

	model.to(device)
	
	if args.summary:
		summary(model, (1, 28, 28))
		print('Batch size: ',  args.batch_size)
		print('Epochs:', args.num_epochs)
	
	trainer = create_supervised_trainer(model, adam_optimizer, cross_entropy_loss, device=device)
	evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy(), "cross": Loss(cross_entropy_loss),
	                                                        "prec": Precision(), "recall": Recall()},
	                                        device=device)
	
	desc = "ITERATION - loss: {:.2f}"
	pbar = tqdm(
		initial=0, leave=False, total=len(train_loader),
		desc=desc.format(0)
	)
	
	
	@trainer.on(Events.ITERATION_COMPLETED)
	def log_training_loss(engine):
		iter = (engine.state.iteration - 1) % len(train_loader) + 1
		
		if iter % 10 == 0:
			pbar.desc = desc.format(engine.state.output)
			pbar.update(10)
	
	
	handler = ModelCheckpoint('/tmp/models', 'fashion-mnist', save_interval=2, n_saved=2, create_dir=Tru
	trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})\
	
	@trainer.on(Events.EPOCH_COMPLETED)
	def log_training_results(engine):
		pbar.refresh()
		evaluator.run(train_loader)
		metrics = evaluator.state.metrics
		avg_accuracy = metrics["accuracy"]
		avg_nll = metrics["cross"]
		writer.add_scalar('TAcc', avg_accuracy, engine.state.epoch)
		writer.add_scalar('TLoss', avg_nll, engine.state.epoch)
		tqdm.write(
			"Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
				.format(engine.state.epoch, avg_accuracy, avg_nll)
		)
	
	
	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(engine):
		evaluator.run(val_loader)
		metrics = evaluator.state.metrics
		avg_accuracy = metrics["accuracy"]
		avg_nll = metrics["cross"]
		writer.add_scalar('VAcc', avg_accuracy, engine.state.epoch)
		writer.add_scalar('VLoss', avg_nll, engine.state.epoch)
		tqdm.write(
			"Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
				.format(engine.state.epoch, avg_accuracy, avg_nll))
		
		pbar.n = pbar.last_print_n = 0
	
	
	trainer.run(train_loader, max_epochs=args.num_epochs)
	pbar.close()
	writer.close()
	
	if args.test_only:
		model.load_state_dict(torch.load(args.test_file)['model_state_dict'])
		model.to(device)
		print('Network loaded')
	tester = create_supervised_evaluator(model, metrics={"accuracy": Accuracy(), "cross": Loss(cross_entropy_loss),
	                                                     "prec": Precision(), "recall": Recall()},
	                                     device=device)
	tester.run(test_loader, max_epochs=1)
	
	@tester.on(Events.EPOCH_COMPLETED)
	def log_training_results(engine):
		pbar.refresh()
		metrics = tester.state.metrics
		avg_accuracy = metrics["accuracy"]
		avg_nll = metrics["cross"]
		prec = metrics["prec"]
		recall = metrics["recall"]
		tqdm.write(
			"Test Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}, Precision {:.2f} Recall: {:.2f},"
				.format(engine.state.epoch, avg_accuracy, avg_nll, prec, recall))


import torch


class CNNForwardBase:
	# def __init__(self, model, loader, optimizer, loss_function, scheduler, device, epochs):
	def __init__(self, model, loader, optimizer, loss_function, device, writer):
		""" Instantiate a CNNForwardBase object
		Args:
			model: This is a torch.nn.Module implemented object
			loader: Dictionary with torch.util.data.DataLoader object as values, keys must be ['train', 'val']
			optimizer: torch.optim type optimizer
			loss_function: any torch criterion function typically, torch.nn.CrossEntropy()
			device: torch.device
			writer: Writer object- Custom or Tensorboard writer
		"""
		
		self.model = model
		self.loader = loader
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.device = device
		self.writer = writer
	
	def train(self, epoch):
		return NotImplementedError
	
	def evaluate(self, epoch):
		return NotImplementedError
	
	def network_pass(self, images, targets):
		outputs = self.model(images)
		loss = self.loss_function(outputs, targets)
		total = targets.size(0)
		_, predicted = torch.max(outputs.data, 1)
		correct = (predicted == targets).sum().item()
		return loss, correct, total


class Trainer(CNNForwardBase):
	def __init__(self, model, loader, optimizer, loss_function, device, writer):
		super(Trainer, self).__init__(model, loader, optimizer, loss_function, device, writer)
		self.model.train()
	
	def train(self, epoch):
		avg_acc = 0
		avg_loss = 0
		with torch.set_grad_enabled(True):
			for batch_id, (train_images, train_labels) in enumerate(self.loader):
				images, targets = train_images.to(self.device), train_labels.to(self.device).long()
				self.optimizer.zero_grad()
				loss, correct, total = self.network_pass(images=images, targets=targets)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				self._adjust_learning_rate(epoch)
				avg_acc += (correct / total) * 100
				avg_loss += loss.item() / total
			self.writer.add_scalar('Train Accuracy', avg_acc / len(self.loader), epoch + 1)
			self.writer.add_scalar('Train Loss', loss.item(), epoch + 1)
		return avg_acc / len(self.loader), avg_loss / len(self.loader)
	
	def _adjust_learning_rate(self, epoch):
		if epoch % 10 == 0:
			lr = 0.001 / 10**(epoch % 10)
			for param_group in self.optimizer.param_groups:
				param_group["lr"] = lr


class Evaluator(CNNForwardBase):
	def __init__(self, model, loader, loss_function, device, writer=None, is_test=False):
		super(Evaluator, self).__init__(model, loader, None, loss_function, device, writer)
		self.model = self.model.eval()
		self.is_test = is_test
	
	def evaluate(self, epoch=None):
		avg_acc = 0
		avg_loss = 0
		with torch.no_grad():
			for batch_id, (images, targets) in enumerate(self.loader):
				images, targets = images.to(self.device), targets.to(self.device).long()
				loss, correct, total = self.network_pass(images, targets)
				avg_acc += (correct / total) * 100
				avg_loss += loss.item() / total
				if not self.is_test:
					self.writer.add_scalar('Val Accuracy', avg_acc / len(self.loader), epoch + 1)
					self.writer.add_scalar('Val Loss', loss.item(), epoch + 1)
		return avg_acc / len(self.loader), avg_loss / len(self.loader)





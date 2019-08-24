import torch


def network_pass(net, images, labels, loss_criterion, optimizer=None, train=False):
	outputs = net(images)
	loss = loss_criterion(outputs, labels)
	# Backprop and perform Adam optimisation
	if train:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	total = labels.size(0)
	_, predicted = torch.max(outputs.data, 1)
	correct = (predicted == labels).sum().item()
	return loss, correct, total


def print_status(loss, correct, total, train, batch_idx=0, epoch=0, num_epochs=0):
	if train:
		print('Epoch [{}/{}], Step {}, Loss: {:.4f}, Accuracy: {:.2f}%'
		      .format(epoch + 1, num_epochs, batch_idx + 1, loss.item(), (correct / total) * 100))
	else:
		print('Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(loss.item(), (correct / total) * 100))


def test(net, device, test_loader, loss_criterion):
	net.eval()
	avg_acc = 0
	avg_loss = 0
	with torch.no_grad():
		for batch_id, (test_images, test_labels) in enumerate(test_loader):
			test_images, test_labels = test_images.to(device), test_labels.to(device).long()
			test_loss, test_correct, test_total = network_pass(net, images=test_images, labels=test_labels,
			                                                   loss_criterion=loss_criterion, train=False)
			avg_acc += test_correct
			avg_loss += test_loss.item()
	print('====================================================')
	print('Test accuracy: {:.2f}%\nTest Loss: {:.2f}'.format(avg_acc / len(test_loader) * 100,
	                                                         avg_loss / len(test_loader)))


def train(net, train_loader, device, optimizer, loss_criterion, num_epochs, epoch, tensorboard=False, writer=None):
	avg_acc = 0
	avg_loss = 0
	net = net.train()
	with torch.set_grad_enabled(True):
		for batch_id, (train_images, train_labels) in enumerate(train_loader):
			train_images, train_labels = train_images.to(device), train_labels.to(device).long()
			optimizer.zero_grad()
			train_loss, train_correct, train_total = network_pass(net, images=train_images, labels=train_labels,
			                                                      optimizer=optimizer, loss_criterion=loss_criterion,
			                                                      train=True)
			if batch_id % 50 == 0:
				print_status( correct=train_correct, total=train_total,
				             num_epochs=num_epochs, batch_idx=batch_id, epoch=epoch, train=True, loss=train_loss)
			avg_acc += (train_correct / train_total) * 100
			avg_loss += train_loss.item()
			if tensorboard:
				writer.add_scalar('Accuracy/train', avg_acc / len(train_loader), epoch)
				writer.add_scalar('Loss/train', avg_loss / len(train_loader), epoch)
	return avg_acc, avg_loss


def evaluate(net, val_loader, device, epoch, loss_criterion, tensorboard=False, writer=None):
	avg_acc = 0
	avg_loss = 0
	net = net.eval()
	with torch.no_grad():
		for batch_id, (val_images, val_labels) in enumerate(val_loader):
			val_images, val_labels = val_images.to(device), val_labels.to(device).long()
			val_loss, val_correct, val_total = network_pass(net, images=val_images, labels=val_labels, train=False,
			                                                loss_criterion=loss_criterion)
			avg_acc += (val_correct / val_total) * 100
			avg_loss = avg_loss * 0.9 + val_loss.item() * 0.1
			if tensorboard:
				writer.add_scalar('Accuracy/Val', avg_acc / len(val_loader), epoch)
				writer.add_scalar('Loss/Val', avg_loss, epoch)
	print_status(total=val_total, correct=val_correct, train=False, loss=val_loss)
	return avg_acc, avg_loss

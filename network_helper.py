from torch.autograd import Variable
import torch


def network_pass(net, images, labels, optimizer, criterion, train=False):
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


def print_status(batch_idx, epoch, loss, correct, total, num_epochs, train):
	if train:
		print('Epoch [{}/{}], Step {}, Loss: {:.4f}, Accuracy: {:.2f}%'
		      .format(epoch + 1, num_epochs, batch_idx + 1, loss.item(), (correct / total) * 100))
	else:
		print('Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(loss.item(), (correct / total) * 100))

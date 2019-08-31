import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

train_transform = transforms.Compose(
	[
		transforms.RandomCrop(28, padding=4, pad_if_needed=False, fill=0, padding_mode='constant'),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	]
)

val_transform = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	]
)

test_transform = transforms.Compose(
	{
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	}
)


def load_data(batch_size, valid_size=0.02):
	train_dataset = torchvision.datasets.FashionMNIST(root='./', train=True, download=True, transform=train_transform)
	val_dataset = torchvision.datasets.FashionMNIST(root='./', train=True, download=False, transform=val_transform)
	test_dataset = torchvision.datasets.FashionMNIST(root='./', train=False, download=True, transform=test_transform)

	num_train = len(train_dataset)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))

	train_indices, val_indices = indices[split:], indices[:split]

	train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2,
	                          sampler=SubsetRandomSampler(train_indices))
	val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, sampler=SubsetRandomSampler(val_indices))
	test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
	return train_loader, val_loader, test_loader

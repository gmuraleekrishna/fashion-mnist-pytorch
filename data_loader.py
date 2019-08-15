import torchvision
from torchvision import transforms
from torch.utils.data import *

transform = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=4, pad_if_needed=False, fill=0, padding_mode='constant'),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ]
)

def load_data():
    train_set = torchvision.datasets.FashionMNIST(root='./', train=True, download=True, transform=transform)
    num_train = len(train_set)
    indices = list(range(num_train))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, sampler=SubsetRandomSampler(indices[59000:]))

    val_set = torchvision.datasets.FashionMNIST(root='./', train=False, download=False, transform=transform)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, sampler=SubsetRandomSampler(indices[:1000]))
    return train_loader, val_loader

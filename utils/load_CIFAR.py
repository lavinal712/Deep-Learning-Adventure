import torchvision
from torch.utils import data
from torchvision import transforms


def load_CIFAR(batch_size, dataset="CIFAR10"):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root='../data', train=True, transform=trans, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root='../data', train=False, transform=trans, download=True)
    elif dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root='../data', train=True, transform=trans, download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            root='../data', train=False, transform=trans, download=True)
    return (data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
            data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4))

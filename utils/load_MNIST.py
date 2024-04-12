import torchvision
from torch.utils import data
from torchvision import transforms


def load_MNIST(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    train_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, transform=trans, download=True)
    return (data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
            data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4))

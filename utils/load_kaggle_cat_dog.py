import torchvision
from torch.utils import data
from torchvision import transforms
from sklearn.model_selection import train_test_split


def load_kaggle_cat_dog(batch_size):
    trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = torchvision.datasets.ImageFolder(root='../data/kagglecatsanddogs_5340/PetImages', transform=trans)
    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [0.8, 0.1, 0.1])
    return (data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
            data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
            data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4))

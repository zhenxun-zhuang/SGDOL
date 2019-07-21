import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

def mnist_data_loader(root_dir, batch_size, valid_ratio=0, **kwargs):
    """
    Utility function for loading train, validation and test iterators
    over the MNIST dataset.

    Arguments:
    - root_dir: root directory to store the dataset.
    - batch_size: how many samples per batch to load.
    - valid_ratio (int, optional): proportion of the training set used for
          validation. Should be in the range [0 - 100]. (default: 0)

    Returns:
    - train_loader: training set iterator.
    - train_size: number of samples in the training set.
    - valid_loader: validation set iterator.
    - valid_size: number of samples in the validation set.
    - test_loader: test set iterator.
    - test_size: number of samples in the test set.
    - classes: names of each class.
    """

    np.random.seed(0)
    
    # Define the transformation which normalizes all images in MNIST.
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    
    # Split the train dataset into train and validation if needed.
    train_dataset = datasets.MNIST(root=root_dir, train=True,
                                   download=True, transform=transform)
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)    

    if valid_ratio > 0:
        valid_dataset = datasets.MNIST(root=root_dir, train=True,
                                       download=True, transform=transform)
    
        split = int(np.floor(valid_ratio / 100.0 * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                  sampler=valid_sampler, shuffle=False,
                                  **kwargs)
        valid_size = len(valid_idx)
    else:
        valid_loader = None
        valid_size = 0
        train_idx = indices

    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler, shuffle=False,
                              **kwargs)
    train_size = len(train_idx)

    test_dataset = datasets.MNIST(root=root_dir, train=False,
                                  download=True, transform=transform) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, **kwargs)
    test_size = len(test_dataset)
    
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    
    return (train_loader, train_size, valid_loader, valid_size,
            test_loader, test_size, classes)
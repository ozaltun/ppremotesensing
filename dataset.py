import torch
from torchvision import datasets, transforms

root = "/Users/boraozaltun/final_project/data/"

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Data Augmentation
def get_standard_data_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(75),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
    }
    return data_transforms


# Get image generators
def get_data_generators(root, data_transforms=None, batch_size=64):

    if data_transforms == None:
        data_transforms = get_standard_data_transforms()
    image_datasets = {x: datasets.ImageFolder(root+x, data_transforms[x]) for x in ['train', 'val']}
    weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # Parameters
    params = {'train': {'batch_size': batch_size, 'num_workers':6, 'sampler':sampler}, 'val':{'batch_size': 4, 'num_workers':6}}

    # Generators
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], **params[x]) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return image_datasets, dataloaders, dataset_sizes, class_names

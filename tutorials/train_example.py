from spaghetti import train
from spaghetti import dataset
import torchvision.transforms.v2 as v2
import torch
import os

# To train, you need to supply two datasets:
# - Dataset from **domain 1**: the images of the domain you want to **translate**
# - Dataset from **domain 2**: the images of the domain you want to be **translated into**

# Here we use two small and very truncated datasets as examples:
# - Domain 1: Ten cropped images from LIVECell (Edlund et al., 2021), a dataset of phase-contrast microscopy.
# - Domain 2: Ten images from PanNuke (Gamper et al., 2020), a dataset of H&E.

# You can find those two truncated dataset at ``./example_datasets/``


domain_1_path_rel = './example_datasets/livecell/'
domain_2_path_rel = './example_datasets/pannuke/'

# get the absolute path
domain_1_path = os.path.abspath(domain_1_path_rel)
domain_2_path = os.path.abspath(domain_2_path_rel)

# we first define the transformations for the images augmentation
transform_L = [v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
            v2.Resize((256, 256)),
            ]
transform = v2.Compose(transform_L)

# prepare the datasets
spagehtti_dataset = dataset.TrainingDataset([domain_1_path], [domain_2_path], 
                                            transform_1=transform, transform_2=transform)

train_dataset, val_dataset = torch.utils.data.random_split(spagehtti_dataset, [0.8, 0.2])

# prepare the dataloaders
# we will use 1 for now for simplicity
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# start training!
train.train_spaghetti(train_loader, val_loader, 
                      name="tutorial_train_outputs", epochs=10)
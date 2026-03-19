from spaghetti import inferences
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

inferences_path_rel = './example_datasets/livecell/'
output_path_rel = './tutorial_translated_images/'

# Train your own model or using the pre-trained model checkpoint downloaded from the link provided in the README
checkpoint_path = './spaghetti_checkpoint.ckpt'

# get the absolute path
inferences_path = os.path.abspath(inferences_path_rel)
output_path = os.path.abspath(output_path_rel)
checkpoint_path = os.path.abspath(checkpoint_path)

# get all the img paths
imgs = []
file_names = []
for path, _, files in os.walk(inferences_path):
    for f in files:
        if f.endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif")):
            imgs.append(os.path.join(path, f))
            file_names.append(f.split(".")[0])

# create the model
model = inferences.Spaghetti(checkpoint_path)

# create a dataset and dataloader for the images
# you can also optionally use a list/tuple to hold all the images and pass that list/tuple directly to the inference function, 
# but using a dataloader is more efficient for large datasets and allows you to perform the pre-processing on the fly
class ImageDataset(Dataset):
    def __init__(self, img_paths, model):
        self.img_paths = img_paths
        self.model = model
        # we need to perform the pre-processing on the images
        # we will use the default transformation, but you can also define your own transformation using a callable
        self.transform = "default"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        transformed_img = self.model.pre_processing([img], transform=self.transform)[0]
        return transformed_img

dataset = ImageDataset(imgs, model)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# perform the inferences, this will return the images in a list of cpu torch.Tensor of each translated image if save_path is None
# otherwise the images will be saved to the output_path and the no images will be returned 
model.inference(dataloader, file_names, output_path)

# you can then do all kinds of fun stuff using H&E models on those translated images!



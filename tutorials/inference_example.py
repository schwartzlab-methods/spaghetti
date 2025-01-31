from spaghetti import inferences
import os
from PIL import Image

inferences_path_rel = './example_datasets/livecell/'
output_path_rel = './tutorial_translated_images/'

# Train your own model or using the pre-trained model checkpoint downloaded from the link provided in the README
checkpoint_path = './spaghetti_checkpoint.ckpt'

# get the absolute path
inferences_path = os.path.abspath(inferences_path_rel)
output_path = os.path.abspath(output_path_rel)
checkpoint_path = os.path.abspath(checkpoint_path)

# get all the imgs
imgs = []
for path, _, files in os.walk(inferences_path):
    for f in files:
        if f.endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif")):
            imgs.append(os.path.join(path, f))

# create the model
model = inferences.Spaghetti(checkpoint_path)
pil_imgs = [Image.open(img).convert("RGB") for img in imgs]

# we need to perform the pre-processing on the images
# we will use the default transformation, but you can also define your own transformation using a callable
processed_imgs = model.pre_processing(pil_imgs, transform="default")

# perform the inferences, this will return the images in a list of cpu torch.Tensor of each translated image
# if an output path is supplied, the translated images will be saved to the output_path
outputs = model.inference(processed_imgs, output_path)

# you can then do all kinds of fun stuff using H&E models on those translated images!



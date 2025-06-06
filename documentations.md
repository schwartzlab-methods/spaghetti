# SPAGHETTI Documentations

## ```SPAGHETTI.dataset```
Set up the dataset to train SPAGHETTI

### ```TrainingDataset```
The dataset class for the SPAGHETTI model training. This class is inherited from ``torch.utils.data.Dataset``
- **args:**
    - ```path_1```: list of strings, the paths to the images in domain 1
    - ```path_2```: list of strings, the paths to the images in domain 2
    - ```transform_1```: the transformation for domain 1 images, in ```torchvision.transforms.v2```
    - ```transform_2```: the transformation for domain 2 images, in ```torchvision.transforms.v2```
    - ```num_sample```: int, optional, the number of images to sample from each domain

## ```SPAGHETTI.train```
Modules for training SPAGHETTI

### ```train_spaghetti```
The function to automatically handle all SPAGHETTI training using ```pytorch_lightning```. 
- **args:**
    - ```train_loader```: the PyTorch Dataloader for the training dataset
    - ```val_loader```: the PyTorch Dataloader for the validation dataset
    - ```batch_size```: int, the batch size for the model, default 1
    - ```weights```: list of floats, the weights for the loss functions in the order of GAN loss, cycle loss, identity loss, and SSIM loss. Default [1.0, 10.0, 5.0, 10.0]
    - ``lr``: float, the learning rate for the model, default 0.0002
    - ```save_dir```: str, the directory to save the model checkpoints and logs. Default current directory
    - ```epochs```: int, the number of epochs to train the model, default 100
    - ```name```: str, the name of the model for the logger, default "my_spaghetti"
    - ```num_nodes```: int, the number of nodes to train the model, default 1
    - ```ngpus_per_node```: int, the number of GPUs per node, default "auto" to use all the available GPUs
- **returns:**
    None
- **size effects:**
    Run the training scripts and save model checkpoints, loss logs, and sampling images to ```save_dir```.

## ```inferences```
Modules for performing inferences with SPAGHETTI

### ```Spaghetti```
The main class housing the architecture of SPAGHETTI for inference.
- **args:**
    - ```model_path```: str, the path to the model checkpoint
- **returns:**
The SPAGHETTI inference model, with the following methods:
    - ```pre_processing```
    Method to pre-process the image for SPAGHETTI transformation. This is a static method.
        - **args:**
            - ```imgs```: list[```PIL.Image``` or ```torch.Tensor``` or ```numpy.ndarray```], the images to perform the pre-processing
            - ```transform```: None, "default", or callable of ```torchvision.transform.v2```, the transformation to perform on the images. If None, no transformation is performed. If "default", the default transformation (converting to ```torch.Tensor`` with range [0,1], normalize with mean and std of 0.5, and resize to (256,256)) is performed.
        - **return:**
            list[```torch.Tensor```], the images after the pre-processing. Each image will have the shape of [C, H, W]
    - ```inference```
    Method to translate the images using the SPAGHETTI model initialized with the model checkpoint.
        - **args:**
            - ```img```: list[```torch.Tensor```], the image(s) to perform the inference. Images have to preprocessed first using the ```pre_processing``` method, each with size [C, H, W] (no batch dimension). 
            - ```names```: list[```str```], the names of the images to be saved
            - ```save_path```: ```str``` or ```None```. If ```str```, images will be saved to the path to after the transformation. If ```None```, transalted images will only be returned but not saved
        - **return:**
            list[```torch.Tensor```], the images after the SPAGHETTI transformaton.
        - **side effects:**
        If ```save_path``` is not ```None```, save the images to the specified path. 
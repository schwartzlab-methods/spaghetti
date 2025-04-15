# SPAGHETTI - <ins>S</ins>SIM-restrained <ins>P</ins>h<ins>a</ins>se Contrast Microscopy <ins>G</ins>AN for <ins>H</ins>&<ins>E</ins> <ins>T</ins>ransla<ins>t</ins>ion of <ins>I</ins>mages
Implementation of the SPAGHETTI method for phase-contrast microscopy images pre-processing so that you can use your favourite H&E model on them.

Read the paper at some_big_journal_websites.

## Installing SPAGHETTI

### Installing using PyPI

SPAGHETTI is available on the Python Package Index (PyPI) to be installed with `pip` directly. To install, run:

``pip install pcm-spaghetti``

### Installing Locally

Alternatively, you may also install SPAGHETTI from the GitHub repository directly. To do that, first create a virtual Python environment and install SPAHETTI locally.

```bash
virtualenv --no-download spaghetti
source spaghetti/bin/activate 
git clone https://github.com/schwartzlab-methods/spaghetti
cd spaghetti
python setup.py sdist bdist_wheel
pip install .
```

## Inferences using SPAGHETTI

An example workflow of how to use SPAGHETTI to convert your phase-contrast microscopy images into H&E-like images can be found at `./tutorials/inference_example.py`. Before running the example code, please ensure that you have cloned the default SPAGHETTI checkpoint file properly located at `./spaghetti_checkpoint.ckpt`. If not, please go to the repository and directly download this checkpoint file.

## Inferences with the CLI tool

Alternatively, you can also run inferences using the CLI interface to perform quick inferences. To do this, after you have installed SPAGHETTI, run:

```bash
python spaghetti --input path_to_directory_with_your_images \
--output path_to_directory_to_save_the_images --checkpoint path_to_the_checkpoint_file
```

The checkpoint file can either be the default checkpoint file (to be downloaded from `./spaghetti_checkpoint.ckpt`), or can be the checkpoint files from your own training (see below for more details on how to train your own SPAGHETTI model).

## Inferences with Docker Image

The CLI inference tool of SPAGHETTI is available as a Docker image so that you do not need to worry about setting up the environment. To use it, ensure you have Docker installed, then run:

``docker pull yinnikun/spaghetti:latest``

Before running the following command, please ensure that all your files (the input image/directory and the model checkpoint) is stored in one directory as we will need to mount this directory in the VM for Docker to run.

Suppose all your input files are stored at `/usr/data/spaghetti_inferences/inputs/` and your model checkpoint is loacted at `/usr/data/spaghetti_inferences/model.ckpt`, and you want to save your results at `/usr/data/spaghetti_inferences/outputs/`, run the following command to performan the inference using Docker:

```bash
docker run --rm -v /usr/data/spaghetti_inferences/:/usr/data/ spaghetti \
--input /usr/data/inputs/ --output /usr/data/outputs/ --checkpoint /usr/data/model.ckpt
```

## Training your own model
You can also train your own model to perform the inferences. See an example code at `./tutorials/train_example.py`







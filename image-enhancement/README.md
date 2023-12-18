# Project 1 Image Enhancement: Single Image Haze Removal

## (Bronte) Sihan Li, Cole Crescas, Karan Shah
## CS 7180
### 2023-09-21

## Environment

Training tasks are run in a Linux environment with a CUDA GPU.

## Requirements

To install dependencies, run:

    pip install pytorch==1.10.2 torchvision torchaudio cudatoolkit==11.3
    pip install -r requirements.txt
    pip install image_dehazer

### Dehaze.ipynb
- The main file for the project. It contains the code for data processing as well as training and testing the dehazeformer model.
- For our testing we configured the environment and processed images from our dataset into training and test sets using an 80-20 split.  
- We then loaded the trained variations of models (-S, -B, -M) based on our dataset size and unfroze the last two layers for fine-tuning.  
- We saw the best results using Dehazeformer-s fine-tuned with 2 layers unfrozen.  
- In addition, we measured the training loss and PSNR through increasing epoch to achieve best results.

### train_test_dehazeformer.py
- Wrapper script for training and testing the different dehazeformer models and write results to disc. This assumes that the `data/a2i2/UAV-train/paired_dehaze/images/` directory and training images have been processed and split into training and validation sets using the `Dehaze.ipynb` notebook.
  
- Usage:
  - For training and testing:
  
        python train_test_dehazeformer.py --model dehazeformer-s

  - For test only on a pretrained model:

        python train_test_dehazeformer.py --model dehazeformer-s --test_only

### simple_dehaze.py
- Simple implementation of dehazing algorithm using boundary constraint on the transmission function.


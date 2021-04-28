# ACGAN

Implementation of ACGAN on CIFAR-10 dataset. 

## Folder Description
- generated_images: include images that were generated for qualitative evaluation
- ht_imgs: synthetic images from different hyperparameter configurations
- lsi: results from latent space interpolation
- model: model checkpoints
- msssim: results from ms-ssim evaluation
- plots: additional plots for loss

## File description
- ACGAN.ipynb: main file for training ACGAN
- inception.ipynb: calculate inception scores
- lsi.ipynb: perform latent space interpolation
- model.py, pruned_layers.py, resnet20.py, train_util.py: helper scripts to load the trained resnet-20 model
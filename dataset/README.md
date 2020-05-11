# Simulated Datasets


This folder stores part of our simulated datasets for paper `Multiple-Boundary Clustering and Prioritization to Promote Neural Network Retraining'.


The ability of some image transformation synthetic operators is weak, i.e., the difference between the synthetic dataset and the original is very small (we observe it from the accuracy difference of the model on the dataset). In order to make the synthetic dataset different from the original one, we compound other operators on some operators with weak ability. The details are as follows.

## 1、MNIST.

Scale: we set the scale coefficient as 0.8.

Translate: we translate the image 3 pixels down and right.

Shear: we set the shear coefficient as 0.4.

Rotate: we first translate the image 2 pixels down and right, then rotate 30 degrees.

Contrast: we first translate the image 3 pixels down and right, then adjust the contrast by coefficient of 1.5.

Brightness: we first translate the image 2 pixels down and right, then add the Brightness by 100.


## 2、CIFAR10.

Scale: we first rotate the image by 15 degrees, then set scale the image with the coefficient as 0.9.

Translate: we first rotate the image by 15 degrees, then translate the image 2 pixels down and right.

Shear: we set the shear coefficient as 0.4.

Rotate: we rotate the image by 15 degrees.

Contrast: we first rotate the image by 15 degrees,, then adjust the contrast by coefficient of 1.2.

Brightness: we add the Brightness by 100.        

## 3、SVHN.

Scale: we first rotate the image by 30 degrees, then set scale the image with the coefficient as 0.8.

Translate: we first rotate the image by 30 degrees, then translate the image 3 pixels down and right.

Shear: we set the shear coefficient as 0.4.

Rotate: we rotate the image by 30 degrees.

Contrast: we first rotate the image by 30 degrees, then adjust the contrast by coefficient of 1.5.

Brightness: we first rotate the image by 30 degrees, then add the Brightness by 50.


## Generating Adversarial Examples

We used the framework by [Ma et al.](https://github.com/xingjunm/lid_adversarial_subspace_detection) to generate various adversarial examples (FGSM, BIM-A, BIM-B, JSMA, and C&W). Please refer to [craft_adv_samples.py](https://github.com/xingjunm/lid_adversarial_subspace_detection/blob/master/craft_adv_examples.py) in the above repository of Ma et al., and put them in the adv directory. 

# Super resolution
Enhance the resolution of images using machine learning. This algorithm enlarges the size of image by adding closely related pixels. 

The code provided so far achieves perceptual distance: 64.7048 in training data and 66.6099 in validation data (lower the score better the accuracy of generated images) within 20 epochs.

Epoch 20/20
156/156 [==============================] - 425s 3s/step - loss: 0.0147 - perceptual_distance: 64.7048 - val_loss: 0.0150 - val_perceptual_distance: 66.6099


### Network Architecture

### GAN (General Adversarial Network)
This is a combination of 2 types of neural network models
* Generator - generates high resolution image from low resolution image
* Discriminator - determines given image is generated or original image

While training, both try to outsmart each other leading to increase in
generated images accuracy.

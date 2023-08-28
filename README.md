# Image Classification

Uses machine learning to classify images. The model is trained on the CIFAR-10 dataset. It expects images to be color and 32x32 pixels.

Uses pytorch and torchvision.

## Training the model

```bash
python3 src/train_model.py
```

## Running the model

```bash
python3 src/run_model.py test_image_ship.png
```

## The CIFAR-10 dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
# image-classification

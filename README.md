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

![cifar_dataset](https://github.com/LeoTheG/image-classification/assets/6187214/47904dad-3e33-4334-a1de-c4e562dbd204)

![cifar_2](https://github.com/LeoTheG/image-classification/assets/6187214/cce80e2a-56c7-486f-be36-25af85b84d27)

![readme_image_classification](https://github.com/LeoTheG/image-classification/assets/6187214/77b9e06d-a803-4585-b6b7-7639a6c2accb)

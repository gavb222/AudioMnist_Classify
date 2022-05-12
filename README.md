# Audiomnist_classify
This project builds networks to classify the AudioMNIST dataset into different digits. In the original paper (https://arxiv.org/pdf/1807.03418.pdf) there are two architectures: AudioNet (time domain) and AlexNet (using magnitude spectrograms as input). AudioNet is implemented here at present, but as this repository is updated, more architectures will be added.

The repository is structured such that networks.py contains the model architectures, train_audionet.py contains the training script for AudioNet, and mnist_dump.py contains code for structuring the dataset appropriately for training. The data is loaded with a modified version of TorchVision's ImageFolder dataset and dataloader classes, functional on .wav and .mp3 files.

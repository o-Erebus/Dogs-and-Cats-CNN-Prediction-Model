
---

# Dogs vs Cats Image Classification

This repository contains code for training a convolutional neural network (CNN) to classify images of dogs and cats using TensorFlow and Keras. It also includes instructions for downloading the dataset and training the model.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Kaggle API (for downloading the dataset)

### Installation

1. Clone this repository to your local machine:

    ```
    git clone https://github.com/your_username/dogs-vs-cats.git
    ```

2. Install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

### Downloading Dataset

Before running the code, you need to download the dataset from Kaggle. Follow these steps:

1. Sign in to your Kaggle account or create one if you don't have it.
2. Go to the [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats) page on Kaggle.
3. Download the dataset by clicking on the "Download" button.
4. Place the downloaded `kaggle.json` file in the root directory of this repository.

### Training Model

The training code is provided in the `train_model.py` script. Follow these steps to train the model:

1. Run the script:

    ```
    python train_model.py
    ```

2. The script will download the dataset, preprocess the images, create and train the model, and save the trained model to disk.

### Testing the Model

To test the trained model on a sample image, you can run the `test_model.py` script:

```
python test_model.py
```

This script will load the trained model from disk and perform inference on a sample image.

## Model Architecture

The model architecture consists of a pre-trained InceptionV3 base followed by additional convolutional and dense layers. The key components of the model architecture include:

- Pre-processing of images using TensorFlow's `ImageDataGenerator`.
- Transfer learning using InceptionV3 as the base model.
- Additional convolutional layers followed by max-pooling.
- Dense layers with ReLU activation and dropout regularization.
- Output layer with sigmoid activation for binary classification.

## Results

After training the model, you can visualize the training and validation accuracy/loss using the provided graphs. These graphs are generated using Matplotlib.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

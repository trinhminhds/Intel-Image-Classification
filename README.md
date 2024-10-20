
# CNN Image Classifier
![Static Badge](https://img.shields.io/badge/Python-3.7-grey?logo=python)
![Static Badge](https://img.shields.io/badge/google-colab-%23F9AB00?logo=googlecolab)
![Static Badge](https://img.shields.io/badge/conda-grey?logo=anaconda)
![Static Badge](https://img.shields.io/badge/Tensorflow-2.x-grey?logo=tensorflow)
![Static Badge](https://img.shields.io/badge/cuda-11.2-grey?logo=nvidia)
![Static Badge](https://img.shields.io/badge/cDNN-8.1-grey?logo=nvidia)

### Convolutional Neural Network (CNN) model for image classification using Tensorflow/Keras.

This project leverages a **Convolutional Neural Network (CNN)** for classifying images across various categories. The model is trained on a custom dataset using the Tensorflow/Keras API.

## File Structure

```
CNN-Image-Classifier/
        ├─ data/
        |   ├─ train/
        |   ├─ test/
        |   ├─ validation/
        ├─ models/
        ├─ scripts/
        |   ├─ train_model.py
        |   ├─ preprocess_data.py
        └─ notebooks/
            ├─ cnn_model_build.ipynb
```

## Data Acquisition
This project assumes you have a dataset prepared in the following structure:

```
data/
    ├─ train/
    ├─ test/
    └─ validation/
```

Each folder (train, test, validation) should contain subfolders for each class. For example, if classifying cats and dogs, your `train` folder will have `cats/` and `dogs/` subfolders with images.

You can either use your own dataset or leverage public datasets available on platforms like **Kaggle**. Here's how you can download a dataset using Kaggle's API:

1. Navigate to your Kaggle account and generate an API token. This will download a file named `kaggle.json`.
2. Place the `kaggle.json` file in the appropriate directory and run:

   ```
   !pip install kaggle
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

3. Once the Kaggle API is set up, download your dataset:

   ```
   !kaggle datasets download -d your-dataset
   ```

Unzip the dataset into the `data/` directory, and you are ready to train the CNN model.

## Model Architecture

The model architecture consists of several layers of:

- Convolutional layers for feature extraction
- MaxPooling layers for downsampling
- Fully connected Dense layers for classification
- Dropout layers to prevent overfitting

The final architecture is as follows:

```
Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Dense -> Output
```

You can experiment with deeper architectures depending on the complexity of your dataset.

## Training the Model

To start training the model, you can use the following script:

```
python scripts/train_model.py --epochs 50 --batch_size 32 --data_dir ./data
```

This will begin the training process and output model metrics such as loss, accuracy, etc. The trained model will be saved in the `models/` directory.

## Evaluation

To evaluate the model on the test set, run:

```
python scripts/evaluate_model.py --data_dir ./data/test
```

This will output the accuracy and loss of the model on unseen data.

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.7+
- Tensorflow 2.x
- CUDA 11.2 (for GPU support)
- cuDNN 8.1

To install the required libraries, use the following:

```
pip install -r requirements.txt
```

## GPU Setup (Optional)

If you have a GPU, you can utilize it for faster training. Ensure CUDA and cuDNN are properly installed. You can verify the GPU setup by running:

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
```

Output should list the available GPUs.

## Results

After training the model, you can visualize the performance metrics, including:

- Training and validation accuracy/loss curves
- Confusion matrix
- Precision, Recall, and F1 scores

## Inference

To make predictions on new images:

```
python scripts/predict.py --image_path ./data/test/example.png
```

# Image_classification_conv2D

Here's a README file for the `conv2D.ipynb` Jupyter Notebook.

### Hospital Patient Information Classification using a Convolutional Neural Network (CNN)

This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of patient information from a hospital. The goal is to categorize patients into different illness groups based on visual data.

#### Project Overview

The core of this project is a CNN model designed to perform image classification. The workflow includes:

1.  **Data Loading and Preprocessing:**

      * Images are loaded from Google Drive, as indicated by the `google.colab` and `drive.mount` commands.
      * `ImageDataGenerator` from Keras is used for **data augmentation** on the training set, which helps the model generalize better by applying random transformations like zooming and shearing to the images.
      * Both training and testing images are **rescaled** to a value between 0 and 1.
      * The images are configured to be grayscale and resized to a target size of 250x230 pixels.

2.  **Model Architecture:**

      * The model is a **`Sequential`** Keras model, a linear stack of layers.
      * It consists of four **`Conv2D`** layers with `relu` activation, each followed by a **`MaxPool2D`** layer to downsample the feature maps.
      * A **`Flatten`** layer converts the 2D feature maps into a 1D vector.
      * The model ends with two **`Dense`** layers. The final `Dense` layer has 8 units with a `softmax` activation function, which is appropriate for a multi-class classification problem with 8 distinct classes of illness.

3.  **Training and Evaluation:**

      * The model is compiled with the **`adam`** optimizer and **`binary_crossentropy`** loss function.
      * It is trained for 30 epochs using the preprocessed training data.
      * **`EarlyStopping`** and **`ReduceLROnPlateau`** callbacks are defined but not used in the training loop provided in the notebook.
      * A **`ModelCheckpoint`** is configured to save the best model weights based on the validation loss, but this is also not implemented in the provided training code.

4.  **Prediction:**

      * After training, the model is saved as `ImageCNN.h5`.
      * The script demonstrates how to load the saved model.
      * An individual image (`1.tif`) is loaded, preprocessed (resized to 250x230 grayscale), and then passed to the model for prediction.

#### Files

  * `conv2D.ipynb`: The main Jupyter Notebook containing the code for the CNN model.
  * `hospital.keras`: The saved model weights (as configured by `ModelCheckpoint`).
  * `ImageCNN.h5`: The final saved model.

#### Dependencies

This project requires the following Python libraries:

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `tensorflow`
  * `scikit-learn`
  * `opencv-python` (`cv2`)

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn opencv-python
```

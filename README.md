# PRODIGY_ML_03

Image classification of dogs and cats by implementing SVM

OVERVIEW

The code aims to implement a Support Vector Machine (SVM) for classifying images of cats and dogs from a Kaggle dataset. The classification is based on Histogram of Oriented Gradients (HOG) features extracted from preprocessed grayscale images. The SVM model is trained and evaluated, and a prediction is demonstrated on a sample image.

DATASET

The training archive comprises 25,000 images of dogs and cats. The project focuses on training the SVM model using these files to enable accurate classification. Subsequently, the trained model predicts labels for test1.zip, differentiating between dogs (1) and cats (0). A part of 25000 images were used for traingi and a part of test1.zip was used for testing.

Dataset: https://www.kaggle.com/c/dogs-vs-cats/data

TECHNOLOGIES USED

1.Python: The primary programming language for the entire solution.

2.NumPy: Library used for numerical operations and data handling, particularly for image data preprocessing.

3.Matplotlib: Library for data and image visualization.

4.scikit-learn: Provides machine learning tools for data preprocessing, model training, and evaluation. Includes modules for SVM, grid search, and performance metrics.

5.OpenCV (cv2): Utilized for reading and processing images.

6.scikit-image: Provides tools for image processing, including the HOG feature extraction.

CODE OVERVIEW

1.Dataset Loading: Images are loaded from the provided dataset path, and corresponding labels (categories: 'cat' and 'dog') are assigned.

2.Image Preprocessing: Each image is converted to grayscale and resized to a fixed dimension (64x64 pixels). HOG features are extracted from the preprocessed images.

3.Train-Test Split: The dataset is split into training and testing sets.

4.SVM Model and Grid Search: A Support Vector Machine (SVM) model is chosen for classification. A grid search is performed to find the best hyperparameters (C, kernel) using cross-validation on the training set.

5.Model Training and Evaluation: The SVM model is trained on the training set with the best hyperparameters. Model accuracy is evaluated on the test set, it coomes out to be approximately 83.17% and a classification report is generated

6.Image Prediction: A function (preprocess_image) is defined to preprocess a single image for prediction. An example image is loaded and preprocessed, and its label is predicted using the trained SVM model. The original and preprocessed images are visualized using Matplotlib.

Acuuracy: ~83.17%
Best parameters: 'C': 10, 'kernel': 'rbf'

CONCLUSION

The developed hand gesture recognition model serves as a foundation for intuitive human-computer interaction and gesture-based control systems. The use of deep learning, specifically CNNs, allows the model to learn intricate patterns in hand gestures, providing a robust and accurate solution for real-world applications.

ACKNOLEDGEMENT

The project is for educational purposes and was created as part of @Prodigy_Infotech Internship

Feel free to explore, modify, or expand upon this project!

If you have any queRIES, suggestions, or feedback, please feel free to email me at Jyoshikavoggu@gmail.com

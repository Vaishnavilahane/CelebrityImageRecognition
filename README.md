
#  Person Classifier using Face Detection and Machine Learning

This project performs person classification based on face images using classical machine learning models. It involves cleaning and preprocessing images using Haarcascade classifiers, training models like SVM,Logisitc Regression and Random Forest, and saving the best model for reuse.
I have restricted classification to only 8 people,
1) Maria Sharapova
2) Serena Williams
3) Virat Kohli
4) Cristiano Ronaldo
5) Lionel Messi
6) Sania Mirza
7) PV Sindhu
8) Mary Kom
---

##  Dataset Structure

- `./dataset/` contains raw images, organized by person name (folder name = label).
- Valid images (with 2 detected eyes) are saved to `./dataset/cropped/`.
- ![detect](https://github.com/user-attachments/assets/6ccfa29e-0b41-4a6b-8b36-069e705801e4)

---

##  Data Cleaning

- Images are passed through a face and eye detection pipeline using OpenCV’s Haarcascade classifiers.
- Only images with **exactly 2 eyes** are cropped and saved.
- This ensures a clean dataset of properly aligned face images.
- ![cropped](https://github.com/user-attachments/assets/a7258310-a3c2-497f-99f4-8d503cb60629)

---

##  Data Preprocessing

- All cropped images are resized and **flattened** into 1D feature vectors.
- Person names are **label-encoded** as integers.
- Features are scaled using **StandardScaler** to prepare for model training.
- ![dict](https://github.com/user-attachments/assets/6e3c091f-ebbc-4b6e-8cda-3f6711cba723)

---

##  Model Building

- Three models are trained using pipelines: **SVM**, **Random Forest**, and **Logistic Regression**.
- **GridSearchCV** is used with 5-fold cross-validation to find the best hyperparameters for each model.
- The best model is chosen based on accuracy score.

---

##  Model Evaluation

- Each model’s performance is evaluated on a validation set.
- A results table summarizes model names, best parameters, and cross-validation scores.
- The best-performing model is selected for deployment.
- ![model evaluation](https://github.com/user-attachments/assets/7de7ea34-64b5-4b04-b65a-5ceae324069f)
- SVM model classification report.
- ![classification_report](https://github.com/user-attachments/assets/2cdc665c-6cac-4898-b626-f9aeaa616b36)
- Confusion matrix
- ![confusionmatrix](https://github.com/user-attachments/assets/c89c20f5-3f32-4672-a5e4-d428c7837078)

---

##  Model Saving

- The best model pipeline (including scaler) is saved using `joblib`.
- A dictionary mapping labels to person names is also saved for decoding predictions.

---

## Flask API Server

A simple Flask backend exposes a `/classify_image` endpoint that accepts image files via POST and returns the predicted person name with confidence scores. It uses the pre-trained ML model for real-time classification.

---

## Technologies used
1. Python
2. Numpy and OpenCV for data cleaning
3. Matplotlib & Seaborn for data visualization
4. Sklearn for model building
5. Jupyter notebook, visual studio code and pycharm as IDE

Install using:

```bash
pip install flask opencv-python scikit-learn matplotlib numpy joblib
```


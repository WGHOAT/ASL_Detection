# 🤟 American Sign Language Detection with CNN

This project focuses on building a deep learning model to recognize American Sign Language (ASL) from static images. The model learns to classify 29 different hand gestures, including the alphabet A–Z and special signs like SPACE, DELETE, and NOTHING.

---

## 🎯 Objective

Build a computer vision model that detects a given ASL input image and outputs the corresponding letter or special gesture.

---

## 🗂️ Dataset

- **Source**: [ASL Alphabet Kaggle Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Classes**: 29 total
  - 26 letters (A–Z)
  - 3 symbols: `SPACE`, `DELETE`, and `NOTHING`
- **Structure**: Training dataset is organized in folders per class.
- **Size**: 
  - Train: 69,600 images
  - Validation: 17,400 images

---

## 🧪 Preprocessing

- All images are resized to **64x64** pixels.
- Pixel values are **rescaled to [0, 1]**.
- Dataset split:
  - **80% training**
  - **20% validation**
- Data fed using `ImageDataGenerator` with flow from directory.

---

## 🧠 Model Architecture

```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(29, activation='softmax')
])
```
1.Optimizer: Adam

2.Loss Function: Categorical Crossentropy

3.Metrics: Accuracy

4.Epochs: 5

5.Batch Size: 64

# 📈 Training Results
| Epoch | Train Accuracy | Val Accuracy | Val Loss |
| ----- | -------------- | ------------ | -------- |
| 1     | 32.8%          | 52.8%        | 1.38     |
| 2     | 78.4%          | 68.5%        | 1.04     |
| 3     | 87.8%          | 72.5%        | 0.89     |
| 4     | 91.2%          | 73.3%        | 1.00     |
| 5     | 92.9%          | 71.9%        | 1.12     |


# 🔍 Example Usage 
```python
predict_image('asl_alphabet_test/asl_alphabet_test/A_test.jpg')
```
# 🧱 Dependencies

tensorflow
numpy
matplotlib
keras
Pillow (for image processing)

# 🚀 Future Improvements
Use transfer learning (e.g., MobileNet, EfficientNet)
Add real-time webcam-based detection
Convert model to TFLite and deploy on Android
Hyperparameter tuning (Dropout rate, learning rate)
Add a confusion matrix to visualize misclassifications





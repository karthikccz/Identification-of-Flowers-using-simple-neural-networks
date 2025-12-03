import numpy as np
from tensorflow.keras.models import load_model
import cv2

# ✅ Load trained model
model = load_model("flower_model.h5")

# ✅ Load saved label classes
label_classes = np.load("label_classes.npy", allow_pickle=True)

# ✅ Load and preprocess test image
img = cv2.imread("C:/Users/karth/OneDrive/Desktop/su.jpg")
img = cv2.resize(img, (224, 224)) / 255.0
img = img.reshape(1, 224, 224, 3)

# ✅ Predict
prediction = model.predict(img)
predicted_index = np.argmax(prediction)
predicted_label = label_classes[predicted_index]
confidence = np.max(prediction) * 100

# ✅ Final Output
print(f"✅ Predicted Flower: {predicted_label}")
print(f"🔎 Confidence: {confidence:.2f}%")

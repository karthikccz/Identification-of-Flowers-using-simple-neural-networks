import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# ✅ IMAGE SETTINGS
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

# ✅ LOAD DATASET
def load_data(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        folder = os.path.join(data_dir, label)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

# ✅ DATA PATH (UPDATE IF NEEDED)
data_path = r"C:\Users\karth\Downloads\archive\flowers"
X, y = load_data(data_path)

# ✅ NORMALIZATION
X = X / 255.0

# ✅ LABEL ENCODING
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# ✅ TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# ✅ DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# ✅ TRANSFER LEARNING MODEL
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(len(np.unique(y)), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ✅ COMPILE
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ TRAIN
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS
)

# ✅ EVALUATION
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Test Accuracy: {acc*100:.2f}%")

# ✅ ACCURACY GRAPH
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




# After training and evaluation in flower_final.py

# ✅ Save trained model
model.save("flower_model.h5")

# ✅ Save label classes (for decoding predictions)
np.save("label_classes.npy", le.classes_)

print("Model and label classes saved successfully!")






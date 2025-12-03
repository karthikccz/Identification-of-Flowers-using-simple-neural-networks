import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# ==============================
# 🔁 LOAD MODEL & LABELS (CACHED)
# ==============================
@st.cache_resource
def load_flower_model():
    try:
        model = load_model("flower_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_label_classes():
    try:
        classes = np.load("label_classes.npy", allow_pickle=True)
        return classes
    except Exception as e:
        st.error(f"Error loading label classes: {e}")
        return None

model = load_flower_model()
label_classes = load_label_classes()

# ==============================
# 🎨 STREAMLIT UI LAYOUT
# ==============================
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Classification System")
st.write("Upload a flower image and the system will predict its class using your trained model.")

# Sidebar info
st.sidebar.header("📊 Model Info")
st.sidebar.write("**Architecture:** MobileNetV2 (Transfer Learning)")
st.sidebar.write("**Input Size:** 224 × 224")
st.sidebar.write("**Test Accuracy:** ~89%")
st.sidebar.write("**Classes:** Daisy, Dandelion, Rose, Sunflower, Tulip")

st.markdown("---")

# ==============================
# 📤 IMAGE UPLOAD SECTION
# ==============================
uploaded_file = st.file_uploader(
    "Upload a flower image (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# ✅ SAME PREPROCESSING AS YOUR CV2 SCRIPT
# ==============================
def preprocess_image(image: Image.Image):
    # Convert PIL → OpenCV format (BGR)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize exactly like your script
    img = cv2.resize(img, (224, 224))

    # Normalize exactly like your script
    img = img / 255.0

    # Reshape exactly like your script
    img = img.reshape(1, 224, 224, 3)

    return img

# ==============================
# 🔍 PREDICTION LOGIC (MATCHING YOUR FILE)
# ==============================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")

    if (model is not None) and (label_classes is not None):
        if st.button("🔮 Predict Flower"):
            with st.spinner("Analyzing image..."):
                img_input = preprocess_image(image)

                # ✅ Predict (EXACT MATCH)
                prediction = model.predict(img_input)
                predicted_index = np.argmax(prediction)
                predicted_label = label_classes[predicted_index]
                confidence = np.max(prediction) * 100

            # ✅ Final Output (EXACT MATCH)
            st.success(f"✅ Predicted Flower: **{predicted_label}**")
            st.write(f"🔎 Confidence: **{confidence:.2f}%**")

            # ✅ Show all class probabilities
            st.markdown("### 📊 Class Probabilities")
            for i, cls in enumerate(label_classes):
                prob = float(prediction[0][i]) * 100
                st.write(f"- **{cls}**: {prob:.2f}%")

    else:
        st.error(
            "Model or label classes not loaded. "
            "Please ensure 'flower_model.h5' and 'label_classes.npy' "
            "are present in the same directory as this app."
        )

else:
    st.info("👆 Upload a flower image to start classification.")

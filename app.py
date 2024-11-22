import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# Title of the app
st.title("OCR Text Detection App")

# Sidebar for language selection
languages = st.sidebar.multiselect(
    "Select OCR Languages",
    ['en', 'es', 'fr'],
    default=['en']
)

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Initialize EasyOCR Reader
        reader = easyocr.Reader(languages, gpu=False)

        # Perform OCR
        result = reader.readtext(image_np)

        if not result:
            st.error("No text detected in the image.")
        else:
            # Display OCR Results
            st.success("Text Detected:")
            for detection in result:
                st.write(f"Text: {detection[1]}")

            # Display image with annotations
            img_with_boxes = image_np.copy()
            for detection in result:
                top_left = tuple(map(int, detection[0][0]))
                bottom_right = tuple(map(int, detection[0][2]))
                text = detection[1]
                img_with_boxes = cv2.rectangle(
                    img_with_boxes, top_left, bottom_right, (0, 255, 0), 2
                )
                img_with_boxes = cv2.putText(
                    img_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            st.image(img_with_boxes, caption="Image with Detected Text", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image to proceed.")

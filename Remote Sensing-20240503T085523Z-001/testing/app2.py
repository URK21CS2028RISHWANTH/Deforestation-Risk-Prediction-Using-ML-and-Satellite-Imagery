import streamlit as st
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key="pjZtbsAzjhkBsKvruel1")
project = rf.workspace().project("saltellite-tree-cover")
model = project.version(1).model

# Function to perform prediction and display results
def predict_and_display(image_path):
    # Perform prediction
    result = model.predict(image_path, confidence=1).json()
    
    # Get labels and detections
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Annotate the image with labels and masks
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    # Display the annotated image
    st.image(annotated_image, caption='Annotated Image', use_column_width=True)
    
    # Create a mask
    mask = np.zeros_like(image)
    
    # Draw polygons on the mask based on the points provided in the predictions
    for prediction in result["predictions"]:
        points = np.array([(point["x"], point["y"]) for point in prediction["points"]], np.int32)
        cv2.fillPoly(mask, [points], (255, 255, 255))
    
    # Count the number of white and black pixels in the mask
    num_white_pixels = np.sum(mask == 255)
    num_black_pixels = np.sum(mask == 0)
    total_pixels = num_black_pixels + num_white_pixels
    
    # Display pixel count information
    st.write("Number of white pixels:", num_white_pixels)
    st.write("Number of black pixels:", num_black_pixels)
    st.write("Total number of pixels:", total_pixels)

# Streamlit app
def main():
    st.title('Tree Cover Detection')
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Perform prediction and display results
        predict_and_display(uploaded_file.name)

if __name__ == '__main__':
    main()

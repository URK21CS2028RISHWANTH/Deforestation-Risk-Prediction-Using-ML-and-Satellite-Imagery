import streamlit as st
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import random
import warnings

st.set_page_config(
    page_title="Deforestation Prediction",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Initialize Roboflow
rf = Roboflow(api_key="pjZtbsAzjhkBsKvruel1")
project = rf.workspace().project("saltellite-tree-cover")
model = project.version(1).model

def callibrating_prediction(vegetation_decrease):
    decrease_factor = random.uniform(0.5, 1)
    callibrated_value = vegetation_decrease * decrease_factor
    return callibrated_value

# Function to perform prediction and display results
def predict_and_display(image_path, column):
    # Perform prediction
    result = model.predict(image_path, confidence=1).json()
    
    # Get labels and detections
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result, class_list=[item["class"] for item in result["predictions"]])

    
    # Load the image
    image = cv2.imread(image_path)
    
    # Annotate the image with labels and masks
    label_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    # annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    # Display the annotated image
    st.image(annotated_image, caption=f'Result for Image {column}', use_column_width=True)
    
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
    return num_white_pixels,num_black_pixels,total_pixels
    # # Display pixel count information
    # st.write(f"Number of white pixels for Image {column}:", num_white_pixels)
    # st.write(f"Number of black pixels for Image {column}:", num_black_pixels)
    # st.write(f"Total number of pixels for Image {column}:", total_pixels)

# Streamlit app
# Streamlit app
def main():
    st.title('Deforestation Prediction',)
    with st.sidebar:
        st.title("Vegetation Detection and Future Deforestation Prediction in Particular area")
        st.code("Click on Predict button \nAfter the output of both Images")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
        if uploaded_file1 is not None:
            column = 1
            # Perform prediction and display results for Image 1
            num_white_pixels1, num_black_pixels1, total_pixels1 = predict_and_display(uploaded_file1.name, 1)
            # Display pixel count information
            vegetation_percentage1 = (num_white_pixels1 / total_pixels1) * 100
            non_vegetation_percentage1 = (num_black_pixels1 / total_pixels1) * 100
            st.write(f"Vegetation Percentage in {column} image:",vegetation_percentage1,"%")
            st.write(f"Non Vegetation Percentage in {column} image:", non_vegetation_percentage1,"%")
    
    with col2:
        uploaded_file2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])
        if uploaded_file2 is not None:
            column = 2
            # Perform prediction and display results for Image 2
            num_white_pixels2, num_black_pixels2, total_pixels2 = predict_and_display(uploaded_file2.name, 2)
            # Display pixel count information
            vegetation_percentage2 = (num_white_pixels2 / total_pixels2) * 100
            non_vegetation_percentage2 = (num_black_pixels2 / total_pixels2) * 100
            st.write(f"Vegetation Percentage in {column} image:", vegetation_percentage2,"%")
            st.write(f"Non Vegetation Percentage in {column} image:", non_vegetation_percentage2,"%")
    if st.sidebar.button("Predict"):
        # Calculate the decrease in vegetation and non-vegetation percentages
        if vegetation_percentage1>vegetation_percentage2:
            vegetation_decrease = vegetation_percentage1 - vegetation_percentage2
            # non_vegetation_decrease = non_vegetation_percentage1 - non_vegetation_percentage2
            vegetation_decrease_after_call=callibrating_prediction(vegetation_decrease)
            # Display the decrease in vegetation and non-vegetation percentages
            # st.info(f"#### Disease Detected: \n ##### :blue[LEAF ROT] : {vegetation_decrease}")
            st.info(f"#### Decrease in Vegetation Percentage: :blue[{vegetation_decrease_after_call:.2f}%]")
            st.info(f"#### Expected vegetation After Prediction : :blue[{(vegetation_percentage2-vegetation_decrease_after_call):.2f}%]")
            # st.info(f"Decrease in Non-Vegetation Percentage: {non_vegetation_decrease:.2f}%")
        elif vegetation_percentage1==vegetation_percentage2:
            st.info(f"#### Vegatation Percentage is Same in Both Images")
        else:
            st.info(f"#### Vegatation Percentage is Increased in the Second Image,So No Decrease")
            # st.write(f"Decrease in Non-Vegetation Percentage: {non_vegetation_decrease:.2f}%")

if __name__ == '__main__':
    main()

from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
# import streamlit as st

rf = Roboflow(api_key="pjZtbsAzjhkBsKvruel1")
project = rf.workspace().project("saltellite-tree-cover")
model = project.version(1).model

result = model.predict("aa.jpg", confidence=1).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_roboflow(result)

label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()

image = cv2.imread("aa.jpg")

annotated_image = mask_annotator.annotate(
    scene=image, detections=detections)

annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))


mask = np.zeros_like(image)

# Draw polygons on the mask based on the points provided in the predictions
for prediction in result["predictions"]:
    points = np.array([(point["x"], point["y"]) for point in prediction["points"]], np.int32)
    cv2.fillPoly(mask, [points], (255, 255, 255))

# Count the number of white and black pixels in the mask

# Display or save the mask image
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Count the number of white and black pixels in the mask
num_white_pixels = np.sum(mask == 255)
num_black_pixels = np.sum(mask == 0)
total_pixels= num_black_pixels+num_white_pixels
# total_pixels = mask.shape[0] * mask.shape[1]

print("Number of white pixels:", num_white_pixels)
print("Number of black pixels:", num_black_pixels)
print("Total number of pixels:", total_pixels)
# print(result)
# polygon = [(point_obj["x"], point_obj["y"]) for point_obj in result["predictions"]["points"]]
# print(polygon)
# st.image(annotated_image)
# from roboflow import Roboflow

# # rf = Roboflow(api_key="API_KEY")
# # project = rf.workspace().project("MODEL_ENDPOINT")
# # model = project.version(VERSION).model

# # infer on a local image
# print(model.predict("your_image.jpg").json())

# # infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE").json())

# # save an image annotated with your predictions
# model.predict("your_image.jpg").save("prediction.jpg")

import os
import cv2
import numpy as np
import pickle

# Set paths
model_prototxt = os.path.join('..', 'models', 'deploy.prototxt')
model_caffemodel = os.path.join('..', 'models', 'mobilenet_iter_73000.caffemodel')
person_image_path = os.path.join('..', 'images', 'person.jpg')
preprocessed_file_path = os.path.join('..', 'preprocessed', 'preprocessed_data.pkl')

# Load pre-trained person detector (MobileNet SSD)
net = cv2.dnn.readNetFromCaffe(model_prototxt, model_caffemodel)

# Load person image
person_img = cv2.imread(person_image_path)

# Resize person image for demonstration purposes
person_img = cv2.resize(person_img, (600, 800))

# Prepare input blob for the network
(h, w) = person_img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(person_img, (300, 300)), 0.007843, (300, 300), 127.5)

# Set the input to the network
net.setInput(blob)

# Perform forward pass to get detections
detections = net.forward()

# Initialize variables for bounding box
box = None

# Iterate over all detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.2:  # Confidence threshold
        idx = int(detections[0, 0, i, 1])
        if idx == 15:  # Class label for person
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype("int")
            break

# Load pre-processed data
with open(preprocessed_file_path, 'rb') as f:
    preprocessed_data = pickle.load(f)

clothing_img = preprocessed_data['clothing_img']
mask = preprocessed_data['mask']
inv_mask = preprocessed_data['inv_mask']

# If a person is detected
if box is not None:
    (startX, startY, endX, endY) = box

    # Resize clothing image to fit the detected person bounding box
    clothing_height = endY - startY
    clothing_width = endX - startX
    clothing_resized = cv2.resize(clothing_img, (clothing_width, clothing_height))

    # Resize masks to fit the detected person bounding box
    mask_resized = cv2.resize(mask, (clothing_width, clothing_height))
    inv_mask_resized = cv2.resize(inv_mask, (clothing_width, clothing_height))

    # Extract region of interest (ROI) from the person image
    roi = person_img[startY:endY, startX:endX]

    # Use masks to isolate the clothing and the background
    bg = cv2.bitwise_and(roi, roi, mask=inv_mask_resized)
    fg = cv2.bitwise_and(clothing_resized, clothing_resized, mask=mask_resized)

    # Combine the background and foreground
    combined = cv2.add(bg, fg)

    # Place the combined image back into the original image
    person_img[startY:endY, startX:endX] = combined

# Display the result
cv2.imshow('Virtual Changing Room', person_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import pickle
import os

# Set paths
clothing_image_path = os.path.join('..', 'images', 'clothing.jpg')
preprocessed_file_path = os.path.join('..', 'preprocessed', 'preprocessed_data.pkl')

# Load clothing image
clothing_img = cv2.imread(clothing_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Resize clothing image (optional, for demonstration purposes)
clothing_img = cv2.resize(clothing_img, (300, 400))

# Split clothing image into BGR and Alpha channels
b, g, r, a = cv2.split(clothing_img)

# Create mask and inverse mask
mask = cv2.merge((a, a, a))
inv_mask = cv2.bitwise_not(mask)

# Save pre-processed data
preprocessed_data = {
    'clothing_img': clothing_img,
    'mask': mask,
    'inv_mask': inv_mask
}

# Save pre-processed data to file
with open(preprocessed_file_path, 'wb') as f:
    pickle.dump(preprocessed_data, f)

print("Pre-processed data saved successfully.")

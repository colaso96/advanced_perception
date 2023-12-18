"""
Cole Crescas & (Bronte) Sihan Li
CS 7180
2023-10-14

This script filters data using color based shadow detection with a shadow threshold. 
We used this to take unfiltered images into folders for testing and shadow removal. 
After running this, the shadow removal model from 
this link https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal can be run for testing.
"""

## File pre-processing

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Specify the input directory containing .ppm images
input_directory = '/content/gdrive/MyDrive/DC-ShadowNet-Hard-and-Soft-Shadow-Removal/Ungrouped_data_ppm'

# Specify the output directories for images with and without shadows
output_directory_with_shadows = '/content/gdrive/MyDrive/DC-ShadowNet-Hard-and-Soft-Shadow-Removal/data/GTSRB_test/Class_Test2/testA'
output_directory_without_shadows = '/content/gdrive/MyDrive/DC-ShadowNet-Hard-and-Soft-Shadow-Removal/data/GTSRB_test/Class_Test2/testB'

# Ensure the output directories exist; create them if necessary
if not os.path.exists(output_directory_with_shadows):
    os.makedirs(output_directory_with_shadows)

if not os.path.exists(output_directory_without_shadows):
    os.makedirs(output_directory_without_shadows)

# List all .ppm files in the input directory
ppm_files = [f for f in os.listdir(input_directory) if f.endswith('.ppm')]

def detect_shadows(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to read the image.")
        return -1

    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Calculate the mean and standard deviation of the L channel
    l_mean, l_stddev = cv2.meanStdDev(l_channel)

    # Set a threshold for detecting shadows based on L channel statistics
    shadow_threshold = l_mean - (l_stddev * 2.0)

    # Create a mask for the shadow pixels
    shadow_mask = l_channel < shadow_threshold

    # Count the number of shadow pixels
    num_shadow_pixels = np.count_nonzero(shadow_mask)

    # Calculate the ratio of shadow pixels to total pixels
    total_pixels = shadow_mask.shape[0] * shadow_mask.shape[1]
    shadow_ratio = num_shadow_pixels / total_pixels

    # Set a threshold for the shadow ratio
    shadow_threshold = 0.01  # You can adjust this threshold as needed
    #print("Shadow ratio: ", round(shadow_ratio, 4))
    # Determine if the image has shadows based on the shadow ratio
    if shadow_ratio > shadow_threshold:
        return True
    else:
        return False

# Loop through each .ppm file, check for shadows, and move to the appropriate folder
for ppm_file in ppm_files:
    ppm_file_path = os.path.join(input_directory, ppm_file)
    
    if detect_shadows(ppm_file_path):
        # Move the image to the "testA" folder
        os.rename(ppm_file_path, os.path.join(output_directory_with_shadows, ppm_file))
    else:
        # Move the image to the "testB" folder
        os.rename(ppm_file_path, os.path.join(output_directory_without_shadows, ppm_file))

print("Shadow detection and categorization complete.")
      
"""
Cole Crescas & (Bronte) Sihan Li
CS 7180
2023-10-14


This script contains multiple functions meant to be used as one off helper functions during data pre-processing.
 Utilities include, moving files out of folders in a directory, changing the type of images and others.
"""

import os
import shutil
from PIL import Image
import cv2
import numpy as np
from google.colab.patches import cv2_imshow


#Helper code to remove files out of a directory


def extract_files_from_subfolders(root_folder, output_folder):
    """
    Extracts all files from subfolders within the root folder and moves them to the output folder.

    Args:
        root_folder (str): The path to the root folder containing subfolders.
        output_folder (str): The path to the folder where extracted files will be placed.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each subfolder in the root folder
    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # Iterate through files in the subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                # Check if the item is a file (not a subdirectory)
                if os.path.isfile(file_path):
                    # Move the file to the output folder
                    shutil.move(file_path, os.path.join(output_folder, filename))

# Example usage:
root_folder = '/content/gdrive/MyDrive/DC-ShadowNet-Hard-and-Soft-Shadow-Removal/data/GTSRB_test/Class_Test'
output_folder = '/content/gdrive/MyDrive/DC-ShadowNet-Hard-and-Soft-Shadow-Removal/Ungrouped_data_ppm'
extract_files_from_subfolders(root_folder, output_folder)

## Helper Code to change the type of an image file

# Specify the directory containing .bmp images
input_directory = '/content/gdrive/MyDrive/DC-ShadowNet-Hard-and-Soft-Shadow-Removal/Ungrouped_data'

# Specify the directory where you want to save the converted .ppm images
output_directory = '/content/gdrive/MyDrive/DC-ShadowNet-Hard-and-Soft-Shadow-Removal/Ungrouped_data_ppm'

# Ensure the output directory exists; create it if necessary
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List all .bmp files in the input directory
bmp_files = [f for f in os.listdir(input_directory) if f.endswith('.bmp')]

# Loop through each .bmp file and convert it to .ppm
for bmp_file in bmp_files:
    # Open the .bmp image
    with Image.open(os.path.join(input_directory, bmp_file)) as img:
        # Construct the output file path with .ppm extension
        ppm_file = os.path.splitext(bmp_file)[0] + '.ppm'

        # Save the image in .ppm format to the output directory
        img.save(os.path.join(output_directory, ppm_file), 'PPM')

print("Conversion complete.")

"""Color-based shadow detection and openCV shadow removal"""



def detect_shadows(image_path, shadow_threshold):
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
    shadow_threshold = 0.02  # You can adjust this threshold as needed
    print("Shadow ratio: ", round(shadow_ratio, 4))
    # Determine if the image has shadows based on the shadow ratio
    if shadow_ratio < shadow_threshold:
        return True
    else:
        return False

def shadow_removal(image_path):
    # Read the color image
    image = cv2.imread(image_path)

    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge the CLAHE-enhanced L channel with the original A and B channels
    enhanced_lab_image = cv2.merge((cl, a_channel, b_channel))

    # Convert the enhanced LAB image back to BGR color space
    result = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    return result


# Test the shadow detection function
images_path = ['/content/test_shadow_image.jpg', '/content/no_shadow_test.jpg', '/content/northeastern_logo_red.jpg',
               '/content/northeastern_logo.jpg', '/content/field_shadow.jpg']
# Set a threshold for the shadow ratio
shadow_threshold = .02

for i in images_path:
  has_shadows = detect_shadows(i, shadow_threshold)

  if has_shadows:
      output_image = shadow_removal(i)
      print("Originial Image")
      cv2_imshow(cv2.imread(i))
      print("Shadow Removed")
      cv2_imshow(output_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

      # Save the processed image
      cv2.imwrite(i[:-4] + "shadow_removed.jpg", output_image)
  else:
      print("The image does not have shadows.")




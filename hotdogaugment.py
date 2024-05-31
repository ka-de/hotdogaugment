"""
    HotdogWolfAug

    This module contains functions for segmenting characters, applying color augmentation,
    and smoothing color transitions in images. It is designed to preprocess and enhance
    images, particularly those containing text or characters, for tasks such as OCR or
    image recognition.

    The module first reads an image, then applies character segmentation to highlight
    characters or text regions in the image. The segmented image is saved to disk.

    Next, the module applies color augmentation to the original image based on the
    segmented regions, creating a color-augmented image. The color-augmented image is
    also saved to disk.

    The module then applies bilateral filtering to both the original and color-augmented
    images to smooth out color transitions. The smoothed images are saved to disk.

    The diameter, sigma_color, and sigma_space parameters for the bilateral filter, as
    well as the value_threshold for the color augmentation, are randomized.

    Finally, the module prints out the execution time for the entire script.
"""

import time
import glob
import os
import random

import cv2
import cv2.ximgproc as ximgproc
import numpy as np

start_time = time.time()


###
def smooth_color_transitions(input_image: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filtering to smooth out color transitions in an image.
    Args:
        input_image (np.ndarray): Input image.
    Returns:
        np.ndarray: The smoothed image.
    """
    diameter = np.random.randint(30, 46)
    sigma_color = np.random.randint(40, 61)
    sigma_space = np.random.randint(80, 101)
    return cv2.bilateralFilter(input_image, diameter, sigma_color, sigma_space)


def segment_characters(input_image: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Segment characters or text regions from an input image.
    Args:
        image (np.ndarray): Input image.
        padding (int, optional): Padding size for the image borders. Default is 20.
    Returns:
        np.ndarray: The segmented image with characters or text regions highlighted.
    """
    input_image = cv2.fastNlMeansDenoisingColored(
        input_image,
        None,
        random.randint(5, 15),
        random.randint(5, 15),
        random.randint(5, 10),
        random.randint(15, 25),
    )

    image_padded = cv2.copyMakeBorder(
        input_image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    gray = cv2.cvtColor(image_padded, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        (random.choice(range(5, 16, 2)), random.choice(range(5, 16, 2))),
        random.randint(10, 20),
    )
    gray = cv2.medianBlur(gray, random.choice(range(3, 10, 2)))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        cv2.approxPolyDP(
            contour, random.uniform(0.2, 0.35) * cv2.arcLength(contour, True), True
        )
        for contour in contours
    ]  # Contour Smoothing
    output = np.zeros_like(thresh)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = thresh[y : y + h, x : x + w]
        output[y : y + h, x : x + w] = cv2.bitwise_and(roi, roi)
    output_no_padding = output[padding:-padding, padding:-padding]
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = random.randint(4, 6)
    eps = random.randint(10, 15)
    output_no_padding = ximgproc.guidedFilter(
        guide, output_no_padding, radius=radius, eps=eps, dDepth=-1
    )
    return output_no_padding


def color_aug(
    input_image: np.ndarray, mask: np.ndarray, value_threshold: int = 50
) -> np.ndarray:
    """
    Apply color augmentation to an image based on a provided mask.
    Args:
        input_image (np.ndarray): Input image.
        mask (np.ndarray): Mask indicating the regions to apply augmentation.
        value_threshold (int, optional): Threshold for brightness value. Default is 50.
    Returns:
        np.ndarray: The color-augmented image.
    """
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    random_hue = np.random.randint(0, 180)
    mask_bright = hsv[:, :, 2] > value_threshold
    hsv[mask != 0 & mask_bright, 0] = (
        hsv[mask != 0 & mask_bright, 0] + random_hue
    ) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Get a list of all .png, .jpg, and .jpeg files
image_files = (
    glob.glob("test-images/*.png")
    + glob.glob("test-images/*.jpg")
    + glob.glob("test-images/*.jpeg")
)

for image_file in image_files:
    print(f"Processing {image_file}...")
    image = cv2.imread(image_file)
    print("Segmenting characters...")
    segmented = segment_characters(image, 50)
    cv2.imwrite(f"segmented_{os.path.basename(image_file)}", segmented)
    print("Applying color augmentation...")
    augmented_image = color_aug(image, segmented, 300)
    print("Smoothing color transitions...")
    smoothed_image = smooth_color_transitions(image)
    smoothed_augmented_image = smooth_color_transitions(augmented_image)
    cv2.imwrite(
        f"smoothed_augmented_{os.path.basename(image_file)}", smoothed_augmented_image
    )
    print(f"Finished processing {image_file}.\n")

end_time = time.time()
execution_time = end_time - start_time
print(f"Executed in {execution_time} seconds")

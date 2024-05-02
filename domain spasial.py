import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Difference Image
def difference_image(image1, image2):
    return cv2.absdiff(image1, image2)
# Average Filter
def average_filter(image, kernel_size=(3, 3)):
    return cv2.blur(image, kernel_size)
# Median Filter
def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)
# Min Filter
def min_filter(image, kernel_size=(3, 3)):
    return cv2.erode(image, np.ones(kernel_size))
# Max Filter
def max_filter(image, kernel_size=(3, 3)):
    return cv2.dilate(image, np.ones(kernel_size))
# Laplacian
def laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)
# Gradient Sobel
def gradient_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_x, sobel_y
# Apply operations
smoothed_image = average_filter(gray_image, kernel_size=(10, 10))
median_filtered = median_filter(gray_image, kernel_size=11)
min_filtered = min_filter(gray_image, kernel_size=(3, 3))
max_filtered = max_filter(gray_image, kernel_size=(3, 3))
laplacian_image = laplacian(gray_image)
sobel_x, sobel_y = gradient_sobel(gray_image)
# Plot results
plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(gray_image, cmap='gray')
plt.show()
plt.subplot(3, 3, 2)
plt.title("Difference Image")
plt.imshow(difference_image(gray_image, smoothed_image), cmap='gray')
plt.show()
plt.subplot(3, 3, 3)
plt.title("Neighborhood Averaging")
plt.imshow(smoothed_image, cmap='gray')
plt.show()
plt.subplot(3, 3, 4)
plt.title("Median Filter")
plt.imshow(median_filtered, cmap='gray')
plt.show()
plt.subplot(3, 3, 5)
plt.title("Min Filter")
plt.imshow(min_filtered, cmap='gray')
plt.show()
plt.subplot(3, 3, 6)
plt.title("Max Filter")
plt.imshow(max_filtered, cmap='gray')
plt.show()
plt.subplot(3, 3, 7)
plt.title("Laplacian")
plt.imshow(laplacian_image, cmap='gray')
plt.show()
plt.subplot(3, 3, 8)
plt.title("Gradient Sobel X")
plt.imshow(sobel_x, cmap='gray')
plt.show()
plt.subplot(3, 3, 9)
plt.title("Gradient Sobel Y")
plt.imshow(sobel_y, cmap='gray')
plt.show()
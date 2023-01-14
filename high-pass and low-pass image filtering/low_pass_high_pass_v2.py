import numpy as np
import cv2
from scipy import signal


def high_pass_function(image):

    # edge detection filter
    kernel = np.array([[-1.0, -1.0, -1.0],
                       [-1.0, 8.0, -1.0],
                       [-1.0, -1.0, -1.0]])

    # kernel = np.array([[-1, 0, 1],
    #                    [-2, 0, 2],
    #                    [-1, 0, 1]])
    #
    # kernel = np.array([[-1, -2, -1],
    #                    [0, 0, 0],
    #                    [1, 2, 1]])

    #
    # # # Sharpen
    # kernel = np.array([[0,  0,  -1, 0,  0],
    #                    [0,  -1, -2, -1, 0],
    #                    [-1, -2, 16, -2, -1],
    #                    [0,  -1, -2, -1, 0],
    #                    [0,  0,  -1, 0,  0]
    #                    ])

    # # Emboss
    # kernel = np.array([[2, 1, 0],
    #                    [1, 0, -1],
    #                    [0, -1, -2]])

    # # Sharpen
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 4, -1],
    #                    [0, -1, 0]])

    convolved_image = signal.convolve2d(image, kernel)

    truncated_image = truncate_v2(convolved_image, kernel)

    high_pass_filtered_image = (truncated_image + image)

    return high_pass_filtered_image


def low_pass_function(image):

    # # Gaussian Filter (Smoothing)
    # kernel = np.array([[1, 2, 1],
    #                    [2, 4, 2],
    #                    [1, 2, 1]]) / 16

    # low pass filter
    kernel = np.ones((3, 3)) / 9

    convolved_image = signal.convolve2d(image, kernel)

    truncated_image = truncate_v2(convolved_image, kernel)

    low_pass_filtered_image = truncated_image

    return low_pass_filtered_image


def truncate_v2(image, kernel):

    m, n = kernel.shape
    m = int((m-1) / 2)

    for i in range(0, m):
        line, row = image.shape
        image = np.delete(image, line-1, 0)
        image = np.delete(image, row-1, 1)
        image = np.delete(image, 0, 0)
        image = np.delete(image, 0, 1)
    return image


def three_channel(image, func):
    return np.stack([func(image[:, :, 0]),  # Channel Blue
                     func(image[:, :, 1]),  # Channel Green
                     func(image[:, :, 2])], axis=2)  # Channel Red

def compare_images(image1_address, image2_address):

    image1 = cv2.imread(image1_address)
    image2 = cv2.imread(image2_address)

    comparison = image1 == image2
    equal_arrays = comparison.all()
    return equal_arrays


# Load the source image
img_src = cv2.imread("src\Atakule.JPG")

# Use high pass filtering the source image then write the image
high_pass_image = three_channel(img_src, high_pass_function)
cv2.imwrite("tmp\high_pass.JPG", high_pass_image)

# Use low pass filtering the source image then write the image
low_pass_image = three_channel(img_src, low_pass_function)
cv2.imwrite("tmp\low_pass.JPG", low_pass_image)

# Compare the images
if (compare_images("tmp\low_pass.JPG", "tmp\high_pass.JPG")):
    print("Images are identically same")
else:
    print("Images are not same")


# First low pass filtering the source image then use high pass filtering the low pass filtered image
high_pass_low_pass_image = three_channel((three_channel(img_src, low_pass_function)), high_pass_function)
cv2.imwrite("tmp\high_pass_low_pass.JPG", high_pass_low_pass_image)

# High pass filtering the source image then use low pass filtering the high pass filtered image
low_pass_high_pass_image = three_channel((three_channel(img_src, high_pass_function)), low_pass_function)
cv2.imwrite("tmp\low_pass_high_pass.JPG", low_pass_high_pass_image)

# Compare the images
if (compare_images("tmp\high_pass_low_pass.JPG", "tmp\low_pass_high_pass.JPG")):
    print("Images are identically same")
else:
    print("Images are not same")

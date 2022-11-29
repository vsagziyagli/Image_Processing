import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

### Gray Level Stretching Funtion
def gray_level_stretching(input, input_start=25, input_end=75, output_start=0, output_end=255):
    if input_start <= input <= input_end:
        output = output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)
        return round(output)
    return round(input)


### Gray Level Stretching Graphic Plot
x_axes = np.linspace(0, 255, num=256)
x_axes = x_axes.tolist()
y_axes = []

for i in x_axes:
    y_axes.append(gray_level_stretching(i))
    print(gray_level_stretching(i), end=" ")

plt.plot(x_axes, y_axes, "--", markersize=1)
plt.xlabel('original gray')
plt.ylabel('stretched gray')
plt.title("Gray-Level Stretching Graph")
plt.savefig("Gray-Level Stretching Graph.jpg")



### Load the Image and Stretch the Gray Level
original_image = 'Linear-ZonePlate.png'

image = cv2.imread(original_image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image_to_np_array = np.array(gray)
original_image_shape = gray_image_to_np_array.shape

counter = 0
pixel_counter = 0
stretched_list = []
for line in gray_image_to_np_array:
    for pixel in line:
        stretched_value = gray_level_stretching(pixel)
        stretched_list.append(stretched_value)

stretched_list_to_np = np.array(stretched_list)
stretched_image_np = np.reshape(stretched_list_to_np, original_image_shape)

stretched_image = Image.fromarray(stretched_image_np)
stretched_image = stretched_image.convert("L")
stretched_image.save("stretched_image.jpeg")


#### Calculate the difference image
difference_np = np.zeros(stretched_image_np.shape)
lines, rows = stretched_image_np.shape
for line in range(lines):
    for pixel in range(rows):
        if stretched_image_np[line][pixel] != gray_image_to_np_array[line][pixel]:
            difference_np[line][pixel] = stretched_image_np[line][pixel]

gray_image = Image.fromarray(gray_image_to_np_array)
gray_image = gray_image.convert("L")
gray_image.save("gray_image.jpeg")

difference_image = Image.fromarray(difference_np)
difference_image = difference_image.convert("L")
difference_image.save("difference_image.jpg")

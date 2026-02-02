# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** R K JAYA KRISNAA  
- **Register Number:** 212223223002

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
gray_img = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()
```

#### 2. Print the image width, height & Channel.
```python
height, width = img_gray.shape
print("Width:", width)
print("Height:", height)
print("Channels: 1 (Grayscale)")

```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight.png', img_gray)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img_color = cv2.imread('Eagle_in_Flight.png')
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
h, w, c = img_color.shape
print("Width:", w)
print("Height:", h)
print("Channels:", c)

plt.imshow(img_color)
plt.title("Color Image")
plt.axis('off')
plt.show()

```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
cropped_img = img_color[100:400, 200:500]
plt.imshow(cropped_img)
plt.title("Cropped Image")
plt.axis('off')
plt.show()
```

#### 8. Resize the image up by a factor of 2x.
```python
resized_img = cv2.resize(cropped_img, None, fx=2, fy=2)
plt.imshow(resized_img)
plt.title("Resized Image")
plt.axis('off')
plt.show()
```

#### 9. Flip the cropped/resized image horizontally.
```python
flipped_img = cv2.flip(resized_img, 1)
plt.imshow(flipped_img)
plt.title("Horizontally Flipped Image")
plt.axis('off')
plt.show()
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
apollo = cv2.imread('Apollo-11-launch.jpg')
apollo = cv2.cvtColor(apollo, cv2.COLOR_BGR2RGB)
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN

h, w, _ = apollo.shape
cv2.putText(apollo, text, (w//6, h-30), font_face, 1.5, (255,255,255), 2)

```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
cv2.rectangle(apollo, (300, 50), (600, 800), (255, 0, 255), 3)
```

#### 13. Display the final annotated image.
```python
plt.imshow(apollo)
plt.title("Apollo 11 Annotated Image")
plt.axis('off')
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
img = cv2.imread('Boy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 15. Adjust the brightness of the image.
```python
matrix_ones = np.ones(img.shape, dtype="float64") * 50

```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img, matrix_ones)
img_darker = cv2.subtract(img, matrix_ones)

```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
titles = ['Original', 'Darker', 'Brighter']
images = [img, img_darker, img_brighter]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.show()
```

#### 18. Modify the image contrast.
```python
matrix1 = np.ones(img.shape) * 1.1
matrix2 = np.ones(img.shape) * 1.2

img_higher1 = cv2.multiply(img, matrix1)
img_higher2 = cv2.multiply(img, matrix2)

```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
titles = ['Original', 'Contrast 1.1', 'Contrast 1.2']
images = [img, img_higher1, img_higher2]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(img)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(b); plt.title('Blue'); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(g); plt.title('Green'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(r); plt.title('Red'); plt.axis('off')
plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```python
merged_rgb = cv2.merge([r, g, b])

plt.imshow(merged_rgb)
plt.title("Merged RGB Image")
plt.axis('off')
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(h); plt.title('Hue'); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(s); plt.title('Saturation'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(v); plt.title('Value'); plt.axis('off')
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.merge([h, s, v])
merged_hsv_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)

plt.imshow(merged_hsv_rgb)
plt.title("Merged HSV Image")
plt.axis('off')
plt.show()
```

## Output:
- **i)** Read and Display an Image.
<Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/da31720f-88be-4276-b3a4-9dc203e2a027" />
  
- **ii)** Draw a line from top-left to bottom-right.
<Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/a8068834-ec7c-48aa-a76e-0b42e6039343" />
<Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/ee3f2ccf-da13-4e0d-821e-eabb0c9dce62" />
<Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/0eada612-62b2-49d9-9e64-61f93e6e302d" />
<Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/c30cef35-f7b7-42ab-8b58-6eb53c9f9e57" />

- **iii)** HVS, GRAY SCALE, YCrCb image.
  <Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/e50991e5-bff6-465c-b363-d8875bbd635c" />
  <Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/36055b75-e6f9-4828-85f8-248c978b8748" />
  <Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/8c34e4cf-ae78-4ae5-a298-10ef75983c20" />
- **iv)** Horizontal and Vertical flip
<Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/e4c29630-b97b-4dba-a813-a60946c9c305" />
<Figure size 640x480 with 1 Axes><img width="515" height="352" alt="image" src="https://github.com/user-attachments/assets/306b266b-cada-42db-88ea-07af86803e55" />

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.


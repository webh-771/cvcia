import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(images, titles, rows, cols, figsize=(15, 8)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for img, title, ax in zip(images, titles, axes):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    for i in range(len(images), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()

image_path = 'chest_xray.jpeg'
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # 1. Preprocessing
    resized_image = cv2.resize(original_image, (512, 512))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.medianBlur(gray_image, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(denoised_image)
    display_images(
        [resized_image, gray_image, denoised_image, enhanced_image],
        ['Resized', 'Grayscale', 'Denoised', 'CLAHE Enhanced'],
        2, 2
    )

    # 2. Color Model Conversion
    enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
    display_images(
        [enhanced_bgr, hsv_image, lab_image],
        ['Enhanced BGR', 'HSV Model', 'LAB Model'],
        1, 3, figsize=(15, 5)
    )

    # 3. Spatial Transformations
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    blur_kernel = np.ones((5, 5), np.float32) / 25
    correlated_sharpened = cv2.filter2D(enhanced_image, -1, sharpen_kernel)
    correlated_blurred = cv2.filter2D(enhanced_image, -1, blur_kernel)
    display_images(
        [correlated_sharpened, correlated_blurred],
        ['Correlation (Sharpen)', 'Correlation (Blur)'],
        1, 2, figsize=(10, 5)
    )

    # 4. Sampling and Quantization
    downsampled_image = cv2.resize(enhanced_image, (128, 128), interpolation=cv2.INTER_NEAREST)
    pixel_values = np.float32(enhanced_image.reshape((-1, 1)))
    K = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()].reshape(enhanced_image.shape)
    display_images(
        [downsampled_image, quantized_image],
        ['Down-Sampled (128x128)', f'Quantized ({K} Levels)'],
        1, 2, figsize=(10, 5)
    )

    # 5. Edge Detection
    sobelx = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    canny_edges = cv2.Canny(enhanced_image, threshold1=50, threshold2=150)
    display_images(
        [sobel_combined, canny_edges],
        ['Sobel Edge Detection', 'Canny Edge Detection'],
        1, 2, figsize=(10, 5)
    )

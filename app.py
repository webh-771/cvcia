import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests

def preprocess_image(image, median_ksize, clahe_clip_limit):
    resized_image = cv2.resize(image, (512, 512))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    if median_ksize % 2 == 0:
        median_ksize += 1
    denoised_image = cv2.medianBlur(gray_image, median_ksize)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(denoised_image)
    return resized_image, denoised_image, enhanced_image

def convert_color_models(enhanced_image):
    enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
    return enhanced_bgr, hsv_image, lab_image

def apply_spatial_transformations(image):
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    blur_kernel = np.ones((5, 5), np.float32) / 25
    correlated_sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    correlated_blurred = cv2.filter2D(image, -1, blur_kernel)
    return correlated_sharpened, correlated_blurred

def sample_and_quantize(image, k_levels):
    downsampled = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
    pixel_values = np.float32(image.reshape((-1, 1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k_levels, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    quantized = np.uint8(centers)[labels.flatten()].reshape(image.shape)
    return downsampled, quantized

def detect_edges(image, canny_thresh1, canny_thresh2):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    canny_edges = cv2.Canny(image, threshold1=canny_thresh1, threshold2=canny_thresh2)
    return sobel_combined, canny_edges

st.set_page_config(layout="wide")
st.title("🫁 Interactive Chest X-Ray Processing Pipeline")
st.write("Upload an image and adjust the parameters in the sidebar to see the pipeline's output in real-time.")

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload an X-ray Image", type=["png", "jpg", "jpeg"])

with st.sidebar.expander("⚙️ Processing Parameters", expanded=True):
    median_ksize = st.slider("Denoising Kernel", 3, 15, 5, step=2, help="Size of the median filter kernel for noise removal.")
    clahe_clip = st.slider("Contrast Limit", 1.0, 10.0, 2.0, 0.5, help="CLAHE clip limit for contrast enhancement.")
    k_levels = st.slider("Quantization Levels", 2, 32, 8, 1, help="Number of gray levels to reduce the image to.")
    canny_t1 = st.slider("Canny Threshold 1", 0, 255, 50, help="Lower threshold for the Canny edge detector.")
    canny_t2 = st.slider("Canny Threshold 2", 0, 255, 150, help="Higher threshold for the Canny edge detector.")

if uploaded_file:
    image = Image.open(uploaded_file)
else:
    url = "https://raw.githubusercontent.com/nihal-21/X-Ray-Image-Processing/main/chest_xray.jpeg"
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        st.sidebar.info("Using a default chest X-ray image.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load default image: {e}")
        st.stop()

original_image = np.array(image)
if len(original_image.shape) == 2:
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
else:
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

st.header("Original Input Image", divider='gray')
st.image(original_image, caption="Original Image", channels="BGR", use_column_width=True)

# --- Pipeline Stages ---
st.header("1. Preprocessing", divider='gray')
st.write("Resizing, removing noise, and enhancing contrast.")
resized_img, denoised_img, enhanced_img = preprocess_image(original_image, median_ksize, clahe_clip)
cols = st.columns(3)
cols[0].image(resized_img, caption="1. Resized & Grayscale", channels="BGR")
cols[1].image(denoised_img, caption=f"2. Denoised (k={median_ksize})")
cols[2].image(enhanced_img, caption=f"3. CLAHE Enhanced (Clip={clahe_clip})")
final_processed_image = enhanced_img

st.header("2. Color Model Conversion", divider='gray')
st.write("Representing brightness in different color spaces.")
enhanced_bgr, hsv, lab = convert_color_models(final_processed_image)
cols = st.columns(3)
cols[0].image(enhanced_bgr, caption="BGR Model", channels="BGR")
cols[1].image(hsv, caption="HSV Model")
cols[2].image(lab, caption="LAB Model")

st.header("3. Spatial Transformations (Correlation)", divider='gray')
st.write("Applying kernels to sharpen and blur the image.")
sharp_img, blur_img = apply_spatial_transformations(final_processed_image)
cols = st.columns(2)
cols[0].image(sharp_img, caption="Sharpened Image")
cols[1].image(blur_img, caption="Blurred Image")

st.header("4. Sampling & Quantization", divider='gray')
st.write("Changing resolution (Sampling) and reducing intensity levels (Quantization).")
downsampled_img, quantized_img = sample_and_quantize(final_processed_image, k_levels)
cols = st.columns(2)
cols[0].image(downsampled_img, caption="Down-Sampled (128x128)")
cols[1].image(quantized_img, caption=f"Quantized to {k_levels} Levels")

st.header("5. Edge Detection", divider='gray')
st.write("Extracting object boundaries by finding high-intensity contrast.")
sobel_edges, canny_edges = detect_edges(final_processed_image, canny_t1, canny_t2)
cols = st.columns(2)
cols[0].image(sobel_edges, caption="Sobel Edge Detection")
cols[1].image(canny_edges, caption="Canny Edge Detection")

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import joblib  # To load RandomForest model

# Load the pre-trained RandomForest model (update this with the actual path)
rf = joblib.load("random_forest_model.pkl")  # Ensure your model is saved as 'random_forest_model.pkl'

# Placeholder for original image
orig_img = None

# Function to select and load an image
def load_image():
    global orig_img
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    
    if file_path:
        orig_img = cv2.imread(file_path)
        if orig_img is None:
            messagebox.showerror("Error", "Unable to load image. Please select a valid image file.")
        else:
            display_image(orig_img)

# Function to display an image in the GUI
def display_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Function for image segmentation (from your code)
def image_segmentation(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_th = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY_INV)
    img_th_inv = 255 - img_th
    return img_th_inv

# Function for image post-processing (dummy function; replace with real logic)
def image_postprocess(img):
    return img

# Placeholder function for extracting features (replace with your logic)
def extract_features(cnt):
    # Return some dummy feature values for now
    return (None, 100, 0.2, 0.8, 0.3, 0.7, 0.6)

# Placeholder function for finding contours (replace with real logic)
def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to process and classify the image
def process_image():
    if orig_img is None:
        messagebox.showerror("Error", "Please load an image first.")
        return

    # Step 1: Image segmentation
    img_seg = image_segmentation(orig_img)

    # Step 2: Image post-processing
    img_pp = image_postprocess(img_seg)

    # Step 3: Find contours (filter contours as per your logic)
    contours_filt = find_contours(img_pp)

    # Prepare dataframe for storing features
    features_list = []

    for cnt in contours_filt:
        features = extract_features(cnt)
        cnt_centre, area, roughness, rel_conv_area, eccentricity, circularity, compactness, eccentricity = features
        cls_lbl = None  
        features_list.append({
            'Area': area, 
            'Roughness': roughness, 
            'Relative convex area': rel_conv_area, 
            'Circularity': circularity, 
            'Compactness': compactness, 
            'Eccentricity': eccentricity, 
            'Class': cls_lbl
        })

    data_test = pd.DataFrame(features_list)
    
    # Step 4: Drop unnecessary features and predict classes using RandomForest
    data_test.drop(["Roughness"], axis=1, inplace=True)
    X_final_test = data_test.iloc[:, :-1]
    y_final_test = rf.predict(X_final_test)
    data_test["Class"] = y_final_test

    # Step 5: Draw bounding boxes and ellipses for visualizing single bacillus and bacilli clusters
    img_bb = orig_img.copy()
    num_sing_tb_list = []
    
    for i, cnt in enumerate(contours_filt):
        (x_ell, y_ell), (ma, MA), angle = cv2.fitEllipse(cnt)
        x_rect, y_rect, w, h = cv2.boundingRect(cnt)
        if data_test.iloc[i, -1] == '1':  # Single bacillus
            cv2.ellipse(img_bb, (int(x_ell), int(y_ell)), (int(ma), int(MA)), int(angle), 0, 360, color=(0, 255, 0), thickness=3)
            num_sing_tb_list.append(1)
        elif data_test.iloc[i, -1] == '2':  # Bacilli cluster
            cv2.rectangle(img_bb, (x_rect, y_rect), (x_rect + w, y_rect + h), (0, 0, 255), 3)
            # Simulate counting bacillus in the cluster
            sing_tb_count, defects = 3, []  # Dummy values, replace with real logic
            num_sing_tb_list.append(max(1, sing_tb_count))
    
    # Step 6: Display results in the GUI
    tb_count.set(f"TB Bacilli detected: {sum(num_sing_tb_list)}")
    display_image(img_bb)

# Initialize Tkinter window
root = tk.Tk()
root.title("TB Bacilli Detection")

# Add a button to load image
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

# Add a button to process image
process_button = tk.Button(root, text="Process Image", command=process_image)
process_button.pack()

# Label to display images
img_label = tk.Label(root)
img_label.pack()

# Label to display TB count
tb_count = tk.StringVar()
tb_count_label = tk.Label(root, textvariable=tb_count)
tb_count_label.pack()

# Main loop to run the Tkinter app
root.mainloop()

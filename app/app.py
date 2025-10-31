from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.morphology import remove_small_objects
import matplotlib
import matplotlib.pyplot as plt
import io
import base64   
from flask import send_file

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

rf = joblib.load('model/rfm.pkl')
img_shape = (600, 800)

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        bacilli_count, plot_image_path, data_test_html, num_sing_tb_list = process_image(filepath)
        
        return render_template('result.html', 
                               image_path=filepath, 
                               count=bacilli_count,
                               plot_image_path=plot_image_path,
                               data_table=data_test_html,
                               num_sing_tb_list=num_sing_tb_list)



def process_image(img_file_path):
    orig_img = cv2.imread(img_file_path)

    if orig_img is None:
        raise ValueError(f"Failed to load image from path {img_file_path}. Ensure the file is an image and is not corrupted.")

    orig_img_shape = orig_img.shape
    img = orig_img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_th = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY_INV)
    img_th_inv = 255 - img_th

    img_seg = image_segmentation(img, orig_img_shape)

    img_shape = (600, 800) 
    img_seg = cv2.resize(img_seg, (img_shape[1], img_shape[0]))

    img_pp = image_postprocess(img_seg)

    # contours, _ = cv2.findContours(img_pp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_filt = find_contours(img_pp, img_shape)

    cols_data = ['Area', 'Relative convex area', 'Circularity', 'Compactness', 'Eccentricity', 'Class']
    # data_test = pd.DataFrame(columns=cols_data)

    img_cnt = np.zeros(img_shape, np.uint8)

    features_list = []

    for cnt in contours_filt:
        features = extract_features(cnt)
        cnt_centre, area, roughness, rel_conv_area, eccentricity, circularity, compactness, eccentricity = features
        cls_lbl = None  
        img_cnt = cv2.drawContours(img_cnt, [cnt], -1, 255, 3)

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
    
    data_test.drop(["Roughness"],axis=1,inplace=True)

    X_final_test = data_test.iloc[:,:-1]

    y_final_test = rf.predict(X_final_test)

    data_test["Class"] = y_final_test

    # print(y_final_test)
    pd.set_option('display.max_rows',None)
    img_bb = img_cnt.copy()  
    for i in range(len(contours_filt)):
        cnt = contours_filt[i]
        (x_ell,y_ell),(ma,MA),angle = cv2.fitEllipse(cnt)
        x_rect,y_rect,w,h = cv2.boundingRect(cnt)
        if (data_test.iloc[i,-1]=='1'):
            #cv2.circle(img_bb,(int(x_ell),int(y_ell)),int(MA),255,3)
            cv2.ellipse(img_bb, (int(x_ell),int(y_ell)), (int(ma),int(MA)), int(angle), 0, 360, color=255, thickness=3) 
        elif (data_test.iloc[i,-1]=='2'):
            cv2.rectangle(img_bb, (x_rect, y_rect), (x_rect+w, y_rect+h), 255, 3)   
    
    num_sing_tb_list = []
    for i in range(data_test.shape[0]):
        sing_tb_count = 0
        if (data_test["Class"][i]=="1"):
            sing_tb_count+=1
        elif (data_test["Class"][i]=="2"):
            cnt = contours_filt[i]
            sing_tb_count,defects = count_single_bacillus(cnt)
            sing_tb_count = max(1,sing_tb_count)
        num_sing_tb_list.append(sing_tb_count)
    
    bacilli_count = sum(num_sing_tb_list)

    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    ax[0, 0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_xlabel("Original Image")
    ax[0, 1].imshow(img_pp, cmap='gray')
    ax[0, 1].set_xlabel("Post processed image")
    ax[1, 0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_xlabel("Original Image")
    ax[1, 1].imshow(cv2.cvtColor(img_bb, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_xlabel("Image with bounding boxes")

    plot_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plot_image.png')
    plt.savefig(plot_image_path)
    plt.close(fig)

    # Convert DataFrame to HTML for rendering
    data_test_html = data_test.to_html()

    return bacilli_count, plot_image_path, data_test_html, num_sing_tb_list


# Function definition of image segmentation to separate the potential bacilli objects from image background

def image_segmentation(img, orig_img_shape):
    orig_img_shape = img.shape
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_cr = img_ycrcb[:,:,1]       
    cr_hist,_ = np.histogram(img_cr.ravel(),256,[0,256])
    cr_hist_diff = np.diff(cr_hist)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_a = img_lab[:,:,1]
    a_hist,_ = np.histogram(img_a.ravel(),256,[0,256])
    a_hist_diff = np.diff(a_hist)
    
    int_lvls = np.arange(0,255)
    
    cr_hist_th = int(-18000*(orig_img_shape[0]*orig_img_shape[1])/(1200*900))
    if (int_lvls[cr_hist_diff<=cr_hist_th].size!=0):
        cr_th_i = np.max(int_lvls[cr_hist_diff<=cr_hist_th])
    else:
        cr_th_i = np.argmax(cr_hist)

    a_hist_th = int(-1000*(orig_img_shape[0]*orig_img_shape[1])/(1200*900))
    if (int_lvls[a_hist_diff<=a_hist_th].size!=0):
        a_th_i = np.max(int_lvls[a_hist_diff<=a_hist_th])
    else:
        a_th_i = np.argmax(a_hist)

    
    a_th = a_th_i
    cr_th = cr_th_i
    _, img_a_th = cv2.threshold(img_a, a_th, 255, cv2.THRESH_BINARY)
    _, img_cr_th = cv2.threshold(img_cr, cr_th, 255, cv2.THRESH_BINARY)
    
    img_seg = cv2.bitwise_and(img_cr_th,img_a_th)

    return img_seg

# Function definition of image postprocessing to remove small size artifacts from segmented image

def image_postprocess(img_seg):
    
    # Removing small size artifacts on segmented image
    img_bin1 = (img_seg//255).astype(bool)
    img_rem1 = remove_small_objects(img_bin1,min_size=20,connectivity=8).astype('uint8')*255
    #img_fill = (remove_small_holes(img_rem1,area_threshold=5000,connectivity=8)).astype('uint8')*255
    #img_bin2 = (img_fill//255).astype(bool)
    #img_rem2 = remove_small_objects(img_bin2,min_size=50,connectivity=8).astype('uint8')*255
    img_pp = img_rem1.copy()
    
    return img_pp

# Function definition to assign class label to each image object

def label_image_objects(img_file,annot_fold_path):

    annot_file = img_file[:-4]+'_annot.csv'
    annot_file_path = os.path.join(annot_fold_path,annot_file)

    annot_cols = ["Label","x_st_pt","y_st_pt","box_width","box_height","Image_name","Image_width","Image_height"]

    lbl_data = pd.read_csv(annot_file_path,names=annot_cols, header=0)
    
    img_lbl = np.ones(orig_img_shape[:2], np.uint8)*20

    for i in range(lbl_data.shape[0]):
        x_st_pt = lbl_data["x_st_pt"][i]
        y_st_pt = lbl_data["y_st_pt"][i]
        x_end_pt = x_st_pt+lbl_data["box_width"][i]
        y_end_pt = y_st_pt+lbl_data["box_height"][i]
        if (lbl_data["Label"][i]=="single bacillus"):
            fill_val = 50
        if (lbl_data["Label"][i]=="bacilli cluster"):
            fill_val = 100
        if (lbl_data["Label"][i]=="unclassified red structures"):
            fill_val = 150
        if (lbl_data["Label"][i]=="artifacts"):
            fill_val = 200
        st_pt = (x_st_pt,y_st_pt)
        end_pt = (x_end_pt,y_end_pt)
        cv2.rectangle(img_lbl, st_pt, end_pt, fill_val, thickness=-1)
    
    img_lbl = cv2.resize(img_lbl,(img_shape[1],img_shape[0]))

    img_pp_lbl = cv2.bitwise_and(img_pp,img_lbl)
    
    return img_lbl,img_pp_lbl

# Function definition to get class label of the contour based on pixel intensity

def get_class_lbl(img_lbl,cnt_centre):
    if (img_lbl[cnt_centre[1],cnt_centre[0]]==50):
        cls_lbl = "1"
    elif (img_lbl[cnt_centre[1],cnt_centre[0]]==100):
        cls_lbl = "2"
    elif (img_lbl[cnt_centre[1],cnt_centre[0]]==20):
        cls_lbl = "3"
    else:
        cls_lbl = None
    
    return cls_lbl

# Function definition to calculate geometric features of a contour

def extract_features(cnt):
    (x,y),(ma,MA),angle = cv2.fitEllipse(cnt)
    cnt_centre = (int(x),int(y))
    eccentricity = ma/MA
    perimeter = cv2.arcLength(cnt,True)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    perimeter_hull = cv2.arcLength(hull,True)
    roughness = perimeter/perimeter_hull
    area_hull = cv2.contourArea(hull)
    rel_conv_area = area_hull/area
    circularity = (4*np.pi*area)/perimeter**2
    compactness = perimeter**2/(4*np.pi*area)
    area_um = area*2.2
    x,y,w,h = cv2.boundingRect(cnt)
    asp_ratio = w/h

    features = (cnt_centre,area,roughness,rel_conv_area,eccentricity,circularity,compactness,eccentricity)
    
    return features

# Function definition to find contours of objects present on the post processed labelled image

def find_contours(img_pp_lbl, img_shape):
    contours,_ = cv2.findContours(img_pp_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_th = 2000
    contours_filt = []
    for cnt in contours:
        if (len(cnt)<5):
            continue
        (x,y),(ma,MA),angle = cv2.fitEllipse(cnt)
        area = cv2.contourArea(cnt)
        area_ell = np.pi*ma*MA/4
        if (x>img_shape[1] or x<0 or y>img_shape[0] or y<0 or area<5 or area>area_th):
            continue
        contours_filt.append(cnt)
        
    return contours_filt      

def count_single_bacillus(cnt):
    sing_tb_count = 0
    cnt_approx = cv2.approxPolyDP(cnt,0.005*cv2.arcLength(cnt, True),True)
    hull = cv2.convexHull(cnt_approx, returnPoints=False)
    hull[::-1].sort(axis=0)
    defects = cv2.convexityDefects(cnt_approx, hull)
    if (defects is not None and defects.shape[0]>1):
        st_ind = -1
        #st_ind = 0
        for j in range(st_ind,defects.shape[0]-1):
            ind1 = defects[j,0,2]
            ind2 = defects[j+1,0,2]
            #print(ind1,ind2)
            if (ind1>ind2):
                list1 = list(range(ind1,cnt_approx.shape[0]))
                list2 = list(range(0,ind2+1))
                list_ = list1+list2
                #print(list_)
                cnt_temp1 = cnt_approx[list_]
            else:
                cnt_temp1 = cnt_approx[ind1:ind2+1]
            #print(cnt_temp1)
            img_temp = np.zeros(img_shape, np.uint8)
            img_temp = cv2.drawContours(img_temp, [cnt_temp1], -1, 255, 3)
            contours_temp,_ = cv2.findContours(img_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_temp2 = contours_temp[0]
            
            if (len(cnt_temp2)>=5):
                (x,y),(ma,MA),angle = cv2.fitEllipse(cnt_temp2)
                eccentricity = ma/MA
                area = cv2.contourArea(cnt_temp2)
                #print((eccentricity,area))
                if (eccentricity>0.1 and eccentricity<0.7 and area>50):
                    sing_tb_count+=1
                    
    return sing_tb_count,defects    

if __name__ == "__main__":
    app.run(debug=True)

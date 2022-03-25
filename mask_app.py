
import numpy as np
import streamlit as st
import cv2
from mtcnn import MTCNN
from PIL import Image, ImageDraw

# https://qiita.com/bear_montblanc/items/e9b47d1cde5088d1c2b0
# https://medium.com/analytics-vidhya/face-detection-webapp-5f947cffbfcb

st.title("マスクをつけましょう★FaceNet’s MTCNN")
imgfile = st.file_uploader("Upload Image", type=["png", "jpg"], accept_multiple_files=False)

if imgfile is not None:    
    pil_img = Image.open(imgfile)
    
    st.write("元の画像")
    st.image(pil_img, use_column_width=True)
    
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = Image.open("mask.png")
    

    # draw = ImageDraw.Draw(img)

    detector = MTCNN()
    results = detector.detect_faces(img)

    for result in results:
        confidence = result["confidence"]
        
        if confidence < 0.9:
            continue
        
        x, y, w, h = result["box"]
        mask_resized = mask.resize((w, h//2))
        pil_img.paste(mask_resized, (x, y+h//2), mask_resized.convert("RGBA"))
        
    pil_img = Image.fromarray(np.uint8(pil_img))
    
    st.write("マスクを付けた画像")
    st.image(pil_img, use_column_width=True)
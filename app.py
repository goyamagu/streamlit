
import numpy as np
import streamlit as st
from mtcnn import MTCNN
from PIL import Image, ImageDraw

st.title("Pythonで顔認識してマスクを付けさせる")
imgfile = st.file_uploader("Upload Image", type=["png", "jpg"], accept_multiple_files=False)

if imgfile is not None:    
    img = Image.open(imgfile)
    mask = Image.open("mask.png")
    
    st.write("元の画像")
    st.image(img, use_column_width=True)

    draw = ImageDraw.Draw(img)

    detector = MTCNN()    
    results = detector.detect_faces(np.asarray(img))

    for result in results:
        confidence = result["confidence"]
        
        if confidence < 0.9:
            continue
        
        x, y, w, h = result["box"]
        mask_resized = mask.resize((w, h//2))        
        img.paste(mask_resized, (x, y+h//2), mask_resized.convert("RGBA"))
        
    pil_img = Image.fromarray(np.uint8(img))
    
    st.write("マスクを付けた画像")
    st.image(pil_img, use_column_width=True)
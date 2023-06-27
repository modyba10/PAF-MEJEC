from flask import request, Flask, render_template, url_for
#~ imports for traitement
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # if len(list(request.files.values()))< 1:
        #     return "Please send in a file <script>setTimeout(() => location.replace(/), 2000)</script>"
        f = list(request.files.values())[0]
        f.save('./static/uploaded_file.png')
        
        #traitement de l'image
        
        img_countoured, contour, symetrie = pipeline2('./static/uploaded_file.png')
        cv2.imwrite('./static/img_contoured.png', img_countoured)
        
        page=render_template("results.html", symetrie=symetrie)
        
        return page
    
    elif request.method == "GET":
        return render_template("webpage.html")
    





#* code de traitement de l'image
#* pipeline 2 : debruitage, dullrazor, égalisation, détection de contours, détection de symétrie
#& debruitage 
def debruitage(image, print=False):
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    
    if (print):
        plt.subplot(121), plt.imshow(image)
        plt.subplot(122), plt.imshow(dst)
        plt.show()
    
    return dst

#& dull razor
def dullrazor(img, print=False):
    # Gray scale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Black hat filter
    kernel = cv2.getStructuringElement(1, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    # Binary thresholding (MASK)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    # Replace pixels of the mask
    dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)
    
    if (print):
    #Display images
        cv2.imshow("Original image", image)
        cv2.imshow("Cropped image", img)
        cv2.imshow("Gray Scale image", grayScale)
        cv2.imshow("Blackhat", blackhat)
        cv2.imshow("Binary mask", mask)
        cv2.imshow("Clean image", dst)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return dst

# & egalisation d'histogramme
def equalizer(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 0] = cv2.equalizeHist(img_hsv[:, :, 0])
    image_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    return image_output

#& detection contours
def detection_contours(img):
    # Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Appliquer un seuillage adaptatif pour améliorer les contours
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Réduire le bruit avec une ouverture morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Créer une copie de l'image pour dessiner les contours
    img_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Dessiner les contours sur l'image
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    return img_contours, contours

#& detection de symétrie
def detection_symetrie(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret,thresh = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_cnt = max(contours, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(max_cnt)
    ellipse_pnts = cv2.ellipse2Poly( (int(ellipse[0][0]),int(ellipse[0][1]) ) ,( int(ellipse[1][0]),int(ellipse[1][1]) ),int(ellipse[2]),0,360,1)
    comp = cv2.matchShapes(max_cnt,ellipse_pnts,1,0.0)
    
    if (comp < 0.099):
        symmetric=True
    else:
        symmetric=False
    
    return symmetric

#* fonction pipeline
def pipeline2 (path):
    image = cv2.imread(path)
    
    img_debruitee = debruitage(image)
    img_razored = dullrazor(img_debruitee)
    img_equalized = equalizer(img_razored)
    img_countoured, contour = detection_contours(img_equalized)
    symetrie = detection_symetrie(img_equalized)
    
    return img_countoured, contour, symetrie

##mettre le ML ici

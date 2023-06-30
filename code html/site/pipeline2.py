# code de traitement de l'image

#~ imports pretraitement
import numpy as np
import cv2
from matplotlib import pyplot as plt
#~ imports compression
from PIL import Image

# * pipeline 2 : debruitage, dullrazor, égalisation, détection de contours, détection de symétrie

# image import
image_path = 'truc'
image = cv2.imread(image_path)

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
    image_path = path
    image = cv2.imread(image_path)
    
    img_debruitee = debruitage(image)
    img_razored = dullrazor(img_debruitee)
    img_equalized = equalizer(img_razored)
    img_countoured, contour = detection_contours(img_equalized)
    symetrie = detection_symetrie(img_countoured)
    
    return img_countoured, contour, symetrie


#* fonction compresion

def compress_jpeg(image, output_image_path, quality):
    #quality between 1 and 95
    
    # l'image est deja en memoire
    # # # Ouvrir l'image en utilisant Pillow
    # # image = Image.open(input_image_path)

    # Convertir l'image en mode RVB si elle n'est pas déjà en RVB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Effectuer la compression JPEG en enregistrant l'image avec la qualité spécifiée
    image.save(output_image_path, format="JPEG", quality=quality,subsampling=1)

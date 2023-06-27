import numpy as np
import cv2
import matplotlib.pyplot as plt

sansillum=cv2.imread("C:/tout/Cours/telecom/PAF/essai1.jpg", cv2.IMREAD_UNCHANGED)
sansillum_bw = cv2.imread("C:/tout/Cours/telecom/PAF/essai1.jpg", 0)
avecillum = cv2.imread("C:/tout/Cours/telecom/PAF/essai2.jpg")
avecillum_bw = cv2.imread("C:/tout/Cours/telecom/PAF/essai2.jpg", 0)

####### histogrammes methode1

figure, axis = plt.subplots(2)
# #sans illum
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([sansillum],[i],None,[256],[0,256])
#     #axis[0].plot(histr,color = col)
#     plt.xlim([0,256])
# histr_bw = cv2.calcHist([sansillum_bw],[0],None,[256],[0,256])
# axis[0].plot(histr_bw, color ="black")



# ## avec illum
# for i,col in enumerate(color):
#     histr = cv2.calcHist([avecillum],[i],None,[256],[0,256])
#     #axis[1].plot(histr,color = col)
#     plt.xlim([0,256])
# histr_bw = cv2.calcHist([avecillum_bw],[0],None,[256],[0,256])
# axis[1].plot(histr_bw, color ="black")

# # axis[2].plot(sansillum_bw)
# # axis[3].plot(avecillum_bw)

# axis[0].set_title("sans flash")
# axis[1].set_title("avec flash")
# # axis[2].set_title("sans flash b&w")
# # axis[3].set_title("avec flash b&w")

# #plt.show()



# #trace des 3 canaux separement
# b,g,r = cv2.split(sansillum)
# cv2.imshow("red", r)
# cv2.imshow("blue", b)
# cv2.imshow("green", g)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #### histogrammes methode 2

# hist,bins = np.histogram(sansillum.flatten(), 256, [0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# axis [0].hist(sansillum.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# #plt.show()



##YUV
img_yuv = cv2.cvtColor(avecillum, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Histogram equalized, YUV', img_output)


##HSV
img_hsv = cv2.cvtColor(avecillum, cv2.COLOR_BGR2HSV)
img_hsv[:,:,0] = cv2.equalizeHist(img_hsv[:,:,0])
img_output2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('Histogram equalized, HSV', img_output2)


##YCrCb
img_YC = cv2.cvtColor(avecillum, cv2.COLOR_BGR2YCrCb)
img_YC[:,:,0] = cv2.equalizeHist(img_YC[:,:,0])
img_output3 = cv2.cvtColor(img_YC, cv2.COLOR_YCrCb2BGR)

cv2.imshow('Color input image', avecillum)
cv2.imshow('Histogram equalized, YCrCb', img_output2)

cv2.waitKey(0)

fig, ax = plt.subplots(4)
ax[0].plot(cv2.calcHist([img_output], [0], None,[256],[0,256]))
ax[1].plot(cv2.calcHist([img_output2], [0], None,[256],[0,256]))
ax[2].plot(cv2.calcHist([img_output3], [0], None,[256],[0,256]))
plt.show()

def equalizer (image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,0] = cv2.equalizeHist(img_hsv[:,:,0])
    image_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return image_output
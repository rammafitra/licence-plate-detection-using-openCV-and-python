# import library yang diperlukan
import cv2
import numpy as np
import pandas as pd
#membaca image
image = cv2.imread('gambar_mobil.JPG')
#konversi image RGB to gray
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#membuat edge detection
canny_edge = cv2.Canny(gray_image, 170, 200)
# menemukan kontur dari image setelah di edge detection
contours, new  = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]
# inisialisasi kontur plat nomor dan membuat koordinat x,y
contour_with_license_plate = None
license_plate = None
x = None
y = None
w = None
h = None
for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4:
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = gray_image[y:y + h, x:x + w]
            break
#mengambarkan bounding box
bounding_box = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 3) 
# menghilangkan noise dari image yang telah di deteksi
license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
thresh, license_plate = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)

canny_edge_2 = cv2.Canny(license_plate, 170, 200)
inversion = cv2.bitwise_not(canny_edge_2)
threshold_level = 1
coords = np.column_stack(np.where(inversion < threshold_level))
DF = pd.DataFrame(coords) 
DF.to_csv("text.txt", index = None)




#menampilkan infomasi gambar
cv2.imshow("original image", image)
cv2.imshow("image to gray ", gray_image)
cv2.imshow(" canny edge detection", canny_edge)
cv2.imshow("bounding box license plat", bounding_box)
cv2.imshow("getting licence plat only",license_plate)
cv2.imshow("testing_1", canny_edge_2)
cv2.imshow("tetsing_2", inversion)
cv2.waitKey(0)

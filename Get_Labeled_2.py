import cv2
import numpy as np
from time import time

# Encuentra ptos comunes en ambas imagenes ingresando solo un pto comun
def Matchear_pts(xr,yr,xl,yl):

    cv2.circle(labeled_img_R, (int(yr), int(xr)), 1, (0, 0, 255), 2)     #las coordenadas van como me las entrega Obtener_Coord
    cv2.circle(labeled_img, (int(yl), int(xl)), 1, (0, 0, 255), 2)
    cv2.circle(imgR, (int(yr), int(xr)), 1, (0, 0, 255), 2)  # las coordenadas van como me las entrega Obtener_Coord
    cv2.circle(img1, (int(yl), int(xl)), 1, (0, 0, 255), 2)


    if labels_R[xr, yr] ==0 or labels[xl, yl] ==0:
        print("Punto no encontrado")
        return (0, 0)

    PrjPoints_R.append(centroids_R[labels_R[xr][yr]])
    PrjPoints_L.append(centroids[labels[xl][yl]])

    localR = labels_R[xr][yr]
    localL = labels[xl][yl]
    localR2 = labels_R[xr][yr]
    localL2 = labels[xl][yl]
    localR3 = labels_R[xr][yr]
    localL3 = labels[xl][yl]
    localR4 = labels_R[xr][yr]
    localL4 = labels[xl][yl]

    # Busca ptos consecutivos hacia atras en horizontal restando alrededor de 8 pixeles y +-2 en vertical
    for a in range(0, 20):
        marca = 0
        if centroids_R[localR][0] - centroids_R[localR - 1][0] < 10 and centroids[localL][0] - centroids[localL - 1][
            0] < 10:
            PrjPoints_R.append(centroids_R[localR - 1])
            PrjPoints_L.append(centroids[localL - 1])
            cv2.circle(labeled_img_R, (int(centroids_R[localR - 1][0]), int(centroids_R[localR - 1][1])), 1,
                       (255, 255, 255), 2)
            cv2.circle(labeled_img, (int(centroids[localL - 1][0]), int(centroids[localL - 1][1])), 1, (255, 255, 255),
                       2)
            cv2.circle(imgR, (int(centroids_R[localR - 1][0]), int(centroids_R[localR - 1][1])), 1,
                       (255, 255, 255), 2)
            cv2.circle(img1, (int(centroids[localL - 1][0]), int(centroids[localL - 1][1])), 1, (255, 255, 255),
                       2)
            localR = localR - 1
            localL = localL - 1
            xr = int(centroids_R[localR][1])
            yr = int(centroids_R[localR][0])
            xl = int(centroids[localL][1])
            yl = int(centroids[localL][0])
        else:
            for i in range(-3, 3):
                if marca == 1:
                    break
                for j in range(-2, 2):
                    if labels_R[xr][int(centroids_R[localR][0] - (8+i))] != 0 and labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                            labels[xl][int(centroids[localL][0] - (8+i))] != 0 and labels[int(centroids[localL][1] - j)][yl] != 0:

                        PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (8+i))]])
                        PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] - (8+i))]])
                        marca = 1
                        print(labels_R[xr][int(centroids_R[localR][0] - (8 + i))])
                        print("patra", labels[xl][int(centroids[localL][0] - (8 + i))])
                        cv2.circle(labeled_img_R, (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (8+i))]][0]),
                                                   int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (8+i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[xl][int(centroids[localL][0] - (8 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] - (8 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (8 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (8 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[xl][int(centroids[localL][0] - (8 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] - (8 + i))]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[xr][int(centroids_R[localR][0] - (8+i))]
                        localL = labels[xl][int(centroids[localL][0] - (8+i))]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])
                        break

    localR = localR2
    localL = localL2
    xr = int(centroids_R[localR][1])
    yr = int(centroids_R[localR][0])
    xl = int(centroids[localL][1])
    yl = int(centroids[localL][0])

    # Busca ptos consecutivos hacia adelante en horizontal sumando alrededor de 8 pixeles y +-2 en vertical
    for a in range(0, 20):
        marca = 0
        if centroids_R[localR + 1][0] - centroids_R[localR][0] < 10 and centroids[localL + 1][0] - centroids[localL][0] < 10:
            PrjPoints_R.append(centroids_R[localR + 1])
            PrjPoints_L.append(centroids[localL + 1])
            cv2.circle(labeled_img_R, (int(centroids_R[localR + 1][0]), int(centroids_R[localR + 1][1])), 1,
                       (255, 255, 255), 2)
            cv2.circle(labeled_img, (int(centroids[localL + 1][0]), int(centroids[localL + 1][1])), 1, (255, 255, 255),
                       2)
            cv2.circle(imgR, (int(centroids_R[localR + 1][0]), int(centroids_R[localR + 1][1])), 1, (255, 255, 255),
                       2)
            cv2.circle(img1, (int(centroids[localL + 1][0]), int(centroids[localL + 1][1])), 1, (255, 255, 255), 2)
            localR = localR + 1
            localL = localL + 1
            xr = int(centroids_R[localR][1])
            yr = int(centroids_R[localR][0])
            xl = int(centroids[localL][1])
            yl = int(centroids[localL][0])

        else:
            for i in range(-3, 3):
                if marca == 1:
                    break
                for j in range(-2, 2):
                    if labels_R[xr][int(centroids_R[localR][0] + (8 + i))] != 0 and \
                            labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                            labels[xl][int(centroids[localL][0] + (8 + i))] != 0 and \
                            labels[int(centroids[localL][1] - j)][yl] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (8 + i))]])
                        PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] + (8 + i))]])
                        marca = 1
                        print(labels_R[xr][int(centroids_R[localR][0] + (8 + i))])
                        print("palante", labels[xl][int(centroids[localL][0] + (8 + i))])
                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (8 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (8 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[xl][int(centroids[localL][0] + (8 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] + (8 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (8 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (8 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[xl][int(centroids[localL][0] + (8 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] + (8 + i))]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[xr][int(centroids_R[localR][0] + (8 + i))]
                        localL = labels[xl][int(centroids[localL][0] + (8 + i))]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])

                        break
    localR = localR3
    localL = localL3
    xr = int(centroids_R[localR][1])
    yr = int(centroids_R[localR][0])
    xl = int(centroids[localL][1])
    yl = int(centroids[localL][0])

    # Busca ptos consecutivos hacia abajo en vertical sumando alrededor de 8 pixeles y +-2 en horizontal
    for a in range(0, 10):
        marca = 0
        for i in range(-3, 3):
            if marca == 1:
                break
            for j in range(-2, 2):
                if labels_R[int(centroids_R[localR][1] + (8 + i))][yr] != 0 and \
                        labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                        labels[int(centroids[localL][1] + (8 + i))][yl] != 0 and \
                        labels[xl][int(centroids[localL][0] + j)] != 0:
                    PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] + (8 + i))][yr]])
                    PrjPoints_L.append(centroids[labels[int(centroids[localL][1] + (8 + i))][yl]])
                    marca = 1

                    print("abajo", labels[int(centroids[localL][1] + (8 + i))][yl])
                    cv2.circle(labeled_img_R,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] + (8 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] + (8 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(labeled_img,
                               (int(centroids[labels[int(centroids[localL][1] + (8 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] + (8 + i))][yl]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(imgR,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] + (8 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] + (8 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(img1,
                               (int(centroids[labels[int(centroids[localL][1] + (8 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] + (8 + i))][yl]][1])),
                               1, (255, 255, 255), 2)

                    localR = labels_R[int(centroids_R[localR][1] + (8 + i))][yr]
                    localL = labels[int(centroids[localL][1] + (8 + i))][yl]

                    xr = int(centroids_R[localR][1])
                    yr = int(centroids_R[localR][0])
                    xl = int(centroids[localL][1])
                    yl = int(centroids[localL][0])

                    break

    localR = localR4
    localL = localL4
    xr = int(centroids_R[localR][1])
    yr = int(centroids_R[localR][0])
    xl = int(centroids[localL][1])
    yl = int(centroids[localL][0])

    # Busca ptos consecutivos hacia arriba en vertical restando alrededor de 8 pixeles y +-2 en horizontal
    for a in range(0, 10):
        marca = 0
        for i in range(-3, 3):
            if marca == 1:
                break
            for j in range(-2, 2):
                if labels_R[int(centroids_R[localR][1] - (8 + i))][yr] != 0 and \
                        labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                        labels[int(centroids[localL][1] - (8 + i))][yl] != 0 and \
                        labels[xl][int(centroids[localL][0] + j)] != 0:
                    PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] - (8 + i))][yr]])
                    PrjPoints_L.append(centroids[labels[int(centroids[localL][1] - (8 + i))][yl]])
                    marca = 1

                    print("arriba", labels[int(centroids[localL][1] - (8 + i))][yl])
                    cv2.circle(labeled_img_R,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] - (8 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] - (8 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(labeled_img,
                               (int(centroids[labels[int(centroids[localL][1] - (8 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] - (8 + i))][yl]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(imgR,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] - (8 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] - (8 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(img1,
                               (int(centroids[labels[int(centroids[localL][1] - (8 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] - (8 + i))][yl]][1])),
                               1, (255, 255, 255), 2)

                    localR = labels_R[int(centroids_R[localR][1] - (8 + i))][yr]
                    localL = labels[int(centroids[localL][1] - (8 + i))][yl]

                    xr = int(centroids_R[localR][1])
                    yr = int(centroids_R[localR][0])
                    xl = int(centroids[localL][1])
                    yl = int(centroids[localL][0])

                    break

    cv2.imshow('CircleR_L', labeled_img_R)
    cv2.imshow('CircleL_L', labeled_img)
    cv2.imshow('CircleR', imgR)
    cv2.imshow('CircleL', img1)

# Start counting.
start_time = time()

imgR = cv2.imread('Rigth.png')
imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
#imgR = imgR[335:2000, 118:2630]
#cv2.imwrite("Right_Cropped.png", imgR)

img1 = cv2.imread('Left.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
#img1 = img1[335:2000, 240:2900]
#cv2.imwrite("Left_Cropped.png", img1)


th2_R = cv2.adaptiveThreshold(imgR, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY_INV, 13, 2)
th2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY_INV, 13, 2)
cv2.imwrite("labeled_R.png", th2_R)
cv2.imwrite("labeled_L.png", th2)

kernel = np.ones((5,5),np.uint8)
th2_R = cv2.morphologyEx(th2_R, cv2.MORPH_OPEN, kernel)
th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
cv2.imwrite('apertura_R.png', th2_R)
cv2.imwrite("apertura_L.png", th2)

output_R = cv2.connectedComponentsWithStats(th2_R, 8, cv2.CV_32S)
output = cv2.connectedComponentsWithStats(th2, 8, cv2.CV_32S)
print(th2_R.shape)
print(th2.shape)
print(output_R[0])

# The first cell is the number of labels
num_R = output_R[0]
# The second cell is the label matrix
labels_R = output_R[1]
# The third cell is the stat matrix
stats_R = output_R[2]
# The fourth cell is the centroid matrix
centroids_R = output_R[3]

# The first cell is the number of labels
num = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]

# Calculate the image process time.
elapsed_image = time() - start_time
print("image process time: %0.10f seconds." % elapsed_image)

M = 98
N = 15
print (num_R)
print (num)
for i in range(1, num_R + 1):
    pts = np.where(labels_R == i)
    pts2 = np.where(labels == i)
    if len(pts[0]) > M or len(pts[0]) < N:
        labels_R[pts] = 0
    if len(pts2[0]) > M or len(pts2[0]) < N:
        labels[pts] = 0


# Calculate the Supress Area time.
elapsed_area = time() - start_time
print("Area process time: %0.10f seconds." % elapsed_area)

xr = 186    #hay q invertir las coordenadas q retorna Obtener_Coord
yr = 484
xl = 176    #hay q invertir las coordenadas q retorna Obtener_Coord
yl = 461

print(labels_R[xr, yr])
print(labels[xl, yl])
print(num_R)
PrjPoints_R = []
PrjPoints_L = []


np.savez('Etiquetas.npz', labels_R=labels_R, labels=labels, centroids_R=centroids_R, centroids=centroids, num_R=num_R, num=num)

#for i in range(1, 1422):
    #pts = np.where(labels == i)
    #labels[pts] = 0
#for i in range(1440, num + 1):
    #pts = np.where(labels == i)
    #labels[pts] = 0

#for i in range(1, 1470):
    #pts = np.where(labels_R == i)
    #labels_R[pts] = 0
#for i in range(1480, num_R + 1):
    #pts = np.where(labels_R == i)
    #labels_R[pts] = 0


# Map component labels to hue val
label_hue = np.uint8(179 * labels_R / np.max(labels_R))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img_R = cv2.merge([label_hue, blank_ch, blank_ch])
# cvt to BGR for display
labeled_img_R = cv2.cvtColor(labeled_img_R, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img_R[label_hue == 0] = 0


# Map component labels to hue val
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img[label_hue == 0] = 0

cv2.imwrite('CircleR_R.png', labeled_img_R)
cv2.imwrite('CircleL_L.png', labeled_img)

# Calculate the final time.
elapsed_final = time() - start_time
print("Elapsed time: %0.10f seconds." % elapsed_final)


cv2.waitKey()
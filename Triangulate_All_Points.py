import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from time import time

start_time = time()
#Carga de valores de Calibración y Etiquetas
with np.load('Valores_Calibracion.npz') as X:
    cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T, E, F, R1, R2, P1, P2, Q = [X[i] for i in ('cameraMatrix1', 'cameraMatrix2', 'distCoeffs1', 'distCoeffs2', 'R', 'T', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q')]
with np.load('Etiquetas.npz') as X:
    labels_R, labels, centroids_R, centroids, num_R, num = [X[i] for i in ('labels_R', 'labels', 'centroids_R', 'centroids', 'num_R', 'num')]


left_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (2752, 2200), cv2.CV_16SC2)
right_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (2752, 2200), cv2.CV_16SC2)

pts_Left = np.array([(centroids[0][0], centroids[0][1])])
pts_Right = np.array([(centroids_R[0][0], centroids_R[0][1])])

for a in range(1, centroids_R.__len__()):
    pto = centroids_R[a][0]
    pto1 = centroids_R[a][1]
    pts_Conca = np.array([(pto, pto1)])
    pts_Right = np.concatenate((pts_Right, pts_Conca))

for a in range(1, centroids.__len__()):
    pto = centroids[a][0]
    pto1 = centroids[a][1]
    pts_Conca = np.array([(pto, pto1)])
    pts_Left = np.concatenate((pts_Left, pts_Conca))


pts_Left = np.float32(pts_Left[:, np.newaxis, :])
pts_Right = np.float32(pts_Right[:, np.newaxis, :])

pts_und_Left = cv2.undistortPoints(pts_Left, cameraMatrix1, distCoeffs1, R=R1, P=P1 )
pts_und_Right = cv2.undistortPoints(pts_Right, cameraMatrix2, distCoeffs2, R=R2, P=P2)

print(pts_und_Right[302], pts_und_Right[302][0], pts_und_Right[302][0][1])  #[0][1] es el eje horizontal
print(pts_und_Left.__len__(), pts_und_Right.__len__())   #cant de etiquetas por imagen

print(pts_und_Left[802], pts_und_Left[803], pts_und_Left[804], pts_und_Left[805], pts_und_Left[806], pts_und_Left[807])
print(pts_und_Right[302], pts_und_Right[303], pts_und_Right[304], pts_und_Right[305], pts_und_Right[306], pts_und_Right[307], pts_und_Right[308], pts_und_Right[309], pts_und_Right[310], pts_und_Right[311])

r1 = []
l1 = []
L = 1090             #5000-7000:1105,
for l in range(5000, 7000):
    for r in range(5000, 7000):
        if L - 3 < pts_und_Right[r][0][1] < L + 3 and L - 3 < pts_und_Left[l][0][1] < L + 3:
            #print(pts_und_Right[r][0],pts_und_Left[l][0])
            #print(r, l)
            r1.append(r)
            l1.append(l)

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
lFrame = cv2.imread('CircleL_L.png')
rFrame = cv2.imread('CircleR_L.png')

left_img_remap = cv2.remap(lFrame, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
right_img_remap = cv2.remap(rFrame, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)


for i in range(0, len(r1)):
    cv2.circle(right_img_remap, (int(pts_und_Right[r1[i]][0][0]), int(pts_und_Right[r1[i]][0][1])), 1, (0, 0, 255), 2)
    #print("aqui", int(pts_und_Right[r1[i]][0][1]), int(pts_und_Right[i][0][0]))

for i in range(0, len(l1)):
    cv2.circle(left_img_remap, (int(pts_und_Left[l1[i]][0][0]), int(pts_und_Left[l1[i]][0][1])), 1, (0, 0, 255), 2)
    #print("aca", int(pts_und_Left[l1[i]][0][1]), int(pts_und_Left[i][0][0]))

left_img_remap[L, :] = (0, 0, 255)
right_img_remap[L, :] = (0, 0, 255)

cv2.imwrite('winname.jpg', np.hstack([left_img_remap, right_img_remap]))

imgR = cv2.imread('Rigth.png')
imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
img1 = cv2.imread('Left.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)


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

    # Busca ptos consecutivos hacia atras en horizontal restando alrededor de 15 pixeles y +-4 en vertical
    for a in range(0, 2):
        marca = 0
        if centroids_R[localR][0] - centroids_R[localR - 1][0] < 20 and centroids[localL][0] - centroids[localL - 1][
            0] < 20:
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
            for i in range(-7, 7):
                if marca == 1:
                    break
                for j in range(-10, 10):
                    if labels_R[xr][int(centroids_R[localR][0] - (15+i))] != 0 and labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                            labels[xl][int(centroids[localL][0] - (15+i))] != 0 and labels[int(centroids[localL][1] - j)][yl] != 0:

                        PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15+i))]])
                        PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] - (15+i))]])
                        marca = 1
                        print(labels_R[xr][int(centroids_R[localR][0] - (15 + i))])
                        print("patra", labels[xl][int(centroids[localL][0] - (15 + i))])
                        cv2.circle(labeled_img_R, (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15+i))]][0]),
                                                   int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15+i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[xr][int(centroids_R[localR][0] - (15+i))]
                        localL = labels[xl][int(centroids[localL][0] - (15+i))]

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
    for a in range(0, 2):
        marca = 0
        if centroids_R[localR + 1][0] - centroids_R[localR][0] < 20 and centroids[localL + 1][0] - centroids[localL][0] < 20:
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
                for j in range(-4, 4):
                    if labels_R[xr][int(centroids_R[localR][0] + (15 + i))] != 0 and \
                            labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                            labels[xl][int(centroids[localL][0] + (15 + i))] != 0 and \
                            labels[int(centroids[localL][1] - j)][yl] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]])
                        PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]])
                        marca = 1
                        print(labels_R[xr][int(centroids_R[localR][0] + (15 + i))])
                        print("palante", labels[xl][int(centroids[localL][0] + (15 + i))])
                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[xr][int(centroids_R[localR][0] + (15 + i))]
                        localL = labels[xl][int(centroids[localL][0] + (15 + i))]

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

    # Busca ptos consecutivos hacia abajo en vertical sumando alrededor de 20 pixeles y +-5 en horizontal
    for a in range(0, 2):
        marca = 0
        for i in range(-5, 5):
            if marca == 1:
                break
            for j in range(-5, 5):
                if labels_R[int(centroids_R[localR][1] + (15 + i))][yr] != 0 and \
                        labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                        labels[int(centroids[localL][1] + (15 + i))][yl] != 0 and \
                        labels[xl][int(centroids[localL][0] + j)] != 0:
                    PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]])
                    PrjPoints_L.append(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]])
                    marca = 1

                    print("abajo", labels[int(centroids[localL][1] + (15 + i))][yl])
                    cv2.circle(labeled_img_R,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(labeled_img,
                               (int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(imgR,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(img1,
                               (int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][1])),
                               1, (255, 255, 255), 2)

                    localR = labels_R[int(centroids_R[localR][1] + (15 + i))][yr]
                    localL = labels[int(centroids[localL][1] + (15 + i))][yl]

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
    for a in range(0, 2):
        marca = 0
        for i in range(-5, 5):
            if marca == 1:
                break
            for j in range(-5, 5):
                if labels_R[int(centroids_R[localR][1] - (15 + i))][yr] != 0 and \
                        labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                        labels[int(centroids[localL][1] - (15 + i))][yl] != 0 and \
                        labels[xl][int(centroids[localL][0] + j)] != 0:
                    PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]])
                    PrjPoints_L.append(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]])
                    marca = 1

                    print("arriba", labels[int(centroids[localL][1] - (15 + i))][yl])
                    cv2.circle(labeled_img_R,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(labeled_img,
                               (int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(imgR,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(img1,
                               (int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][1])),
                               1, (255, 255, 255), 2)

                    localR = labels_R[int(centroids_R[localR][1] - (15 + i))][yr]
                    localL = labels[int(centroids[localL][1] - (15 + i))][yl]

                    xr = int(centroids_R[localR][1])
                    yr = int(centroids_R[localR][0])
                    xl = int(centroids[localL][1])
                    yl = int(centroids[localL][0])

                    break

    cv2.imshow('CircleR_L', labeled_img_R)
    cv2.imshow('CircleL_L', labeled_img)
    cv2.imshow('CircleR', imgR)
    cv2.imshow('CircleL', img1)
def Matchear_pts_All(xr,yr,xl,yl):
    cv2.circle(labeled_img_R, (int(yr), int(xr)), 1, (0, 0, 255),
               2)  # las coordenadas van como me las entrega Obtener_Coord
    cv2.circle(labeled_img, (int(yl), int(xl)), 1, (0, 0, 255), 2)
    cv2.circle(imgR, (int(yr), int(xr)), 1, (0, 0, 255), 2)  # las coordenadas van como me las entrega Obtener_Coord
    cv2.circle(img1, (int(yl), int(xl)), 1, (0, 0, 255), 2)

    if labels_R[xr, yr] == 0 or labels[xl, yl] == 0:
        print("Punto no encontrado")
        return (0, 0)

    PrjPoints_R.append(centroids_R[labels_R[xr][yr]])
    PrjPoints_L.append(centroids[labels[xl][yl]])

    localR = labels_R[xr][yr]
    localL = labels[xl][yl]
    localR2 = labels_R[xr][yr]
    localL2 = labels[xl][yl]

    # Busca ptos consecutivos hacia abajo en vertical sumando alrededor de 20 pixeles y +-5 en horizontal
    def Abajo(localR, localL, xr, yr, xl, yl):

        for a in range(0, 20):
            marca = 0
            for i in range(-5, 5):
                if marca == 1:
                    break
                for j in range(-5, 5):
                    if labels_R[int(centroids_R[localR][1] + (15 + i))][yr] != 0 and \
                            labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                            labels[int(centroids[localL][1] + (15 + i))][yl] != 0 and \
                            labels[xl][int(centroids[localL][0] + j)] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]])
                        PrjPoints_L.append(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]])
                        marca = 1

                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][0]),
                                    int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][0]),
                                    int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][0]),
                                    int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][0]),
                                    int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[int(centroids_R[localR][1] + (15 + i))][yr]
                        localL = labels[int(centroids[localL][1] + (15 + i))][yl]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])

                        break

    # Busca ptos consecutivos hacia arriba en vertical restando alrededor de 8 pixeles y +-2 en horizontal
    def Arriba(localR, localL, xr, yr, xl, yl):
        for a in range(0, 20):
            marca = 0
            for i in range(-5, 5):
                if marca == 1:
                    break
                for j in range(-5, 5):
                    if labels_R[int(centroids_R[localR][1] - (15 + i))][yr] != 0 and \
                            labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                            labels[int(centroids[localL][1] - (15 + i))][yl] != 0 and \
                            labels[xl][int(centroids[localL][0] + j)] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]])
                        PrjPoints_L.append(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]])
                        marca = 1

                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][0]),
                                    int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][0]),
                                    int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][0]),
                                    int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][0]),
                                    int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[int(centroids_R[localR][1] - (15 + i))][yr]
                        localL = labels[int(centroids[localL][1] - (15 + i))][yl]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])

                        break

    # Busca ptos consecutivos hacia atras en horizontal restando alrededor de 15 pixeles y +-4 en vertical
    for a in range(0, 20):
        marca = 0
        if centroids_R[localR][0] - centroids_R[localR - 1][0] < 20 and centroids[localL][0] - centroids[localL - 1][
            0] < 20:
            PrjPoints_R.append(centroids_R[localR - 1])
            PrjPoints_L.append(centroids[localL - 1])

            Abajo(localR, localL, xr, yr, xl, yl)
            Arriba(localR, localL, xr, yr, xl, yl)

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
            for i in range(-7, 7):
                if marca == 1:
                    break
                for j in range(-10, 10):
                    if labels_R[xr][int(centroids_R[localR][0] - (15 + i))] != 0 and \
                            labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                            labels[xl][int(centroids[localL][0] - (15 + i))] != 0 and \
                            labels[int(centroids[localL][1] - j)][yl] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]])
                        PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]])

                        Abajo(localR, localL, xr, yr, xl, yl)
                        Arriba(localR, localL, xr, yr, xl, yl)

                        marca = 1

                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[xr][int(centroids_R[localR][0] - (15 + i))]
                        localL = labels[xl][int(centroids[localL][0] - (15 + i))]

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

    # Busca ptos consecutivos hacia adelante en horizontal sumando alrededor de 8 pixeles y +-4 en vertical
    for a in range(0, 20):
        marca = 0
        if centroids_R[localR + 1][0] - centroids_R[localR][0] < 20 and centroids[localL + 1][0] - centroids[localL][
            0] < 20:
            PrjPoints_R.append(centroids_R[localR + 1])
            PrjPoints_L.append(centroids[localL + 1])

            Abajo(localR, localL, xr, yr, xl, yl)
            Arriba(localR, localL, xr, yr, xl, yl)


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
                for j in range(-4, 4):
                    if labels_R[xr][int(centroids_R[localR][0] + (15 + i))] != 0 and \
                            labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                            labels[xl][int(centroids[localL][0] + (15 + i))] != 0 and \
                            labels[int(centroids[localL][1] - j)][yl] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]])
                        PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]])

                        Abajo(localR, localL, xr, yr, xl, yl)
                        Arriba(localR, localL, xr, yr, xl, yl)

                        marca = 1

                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[xr][int(centroids_R[localR][0] + (15 + i))]
                        localL = labels[xl][int(centroids[localL][0] + (15 + i))]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])

                        break
def Matchear_pts_Arriba(xr,yr,xl,yl):

    localR = labels_R[xr][yr]
    localL = labels[xl][yl]

    if labels_R[xr, yr] ==0 or labels[xl, yl] ==0:
        print("Punto no encontrado")
        return (0, 0)

    PrjPoints_R.append(centroids_R[labels_R[xr][yr]])
    PrjPoints_L.append(centroids[labels[xl][yl]])

    for a in range(0, 40):
        marca = 0
        for i in range(-5, 5):
            if marca == 1:
                break
            for j in range(-5, 5):
                if labels_R[int(centroids_R[localR][1] - (15 + i))][yr] != 0 and \
                        labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                        labels[int(centroids[localL][1] - (15 + i))][yl] != 0 and \
                        labels[xl][int(centroids[localL][0] + j)] != 0:
                    PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]])
                    PrjPoints_L.append(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]])
                    marca = 1

                    print("arriba", labels[int(centroids[localL][1] - (15 + i))][yl])
                    cv2.circle(labeled_img_R,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(labeled_img,
                               (int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(imgR,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(img1,
                               (int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][1])),
                               1, (255, 255, 255), 2)

                    localR = labels_R[int(centroids_R[localR][1] - (15 + i))][yr]
                    localL = labels[int(centroids[localL][1] - (15 + i))][yl]

                    xr = int(centroids_R[localR][1])
                    yr = int(centroids_R[localR][0])
                    xl = int(centroids[localL][1])
                    yl = int(centroids[localL][0])

                    break
def Matchear_pts_Adelante(xr, yr, xl, yl):

    localR = labels_R[xr][yr]
    localL = labels[xl][yl]

    if labels_R[xr, yr] ==0 or labels[xl, yl] ==0:
        print("Punto no encontrado")
        return (0, 0)

    PrjPoints_R.append(centroids_R[labels_R[xr][yr]])
    PrjPoints_L.append(centroids[labels[xl][yl]])

    for a in range(0, 40):
        marca = 0
        if centroids_R[localR + 1][0] - centroids_R[localR][0] < 20 and centroids[localL + 1][0] - centroids[localL][
            0] < 20:
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
                for j in range(-4, 4):
                    if labels_R[xr][int(centroids_R[localR][0] + (15 + i))] != 0 and \
                            labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                            labels[xl][int(centroids[localL][0] + (15 + i))] != 0 and \
                            labels[int(centroids[localL][1] - j)][yl] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]])
                        PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]])
                        marca = 1
                        print(labels_R[xr][int(centroids_R[localR][0] + (15 + i))])
                        print("palante", labels[xl][int(centroids[localL][0] + (15 + i))])
                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[xr][int(centroids_R[localR][0] + (15 + i))]
                        localL = labels[xl][int(centroids[localL][0] + (15 + i))]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])

                        break
def Matchear_pts_Atras(xr, yr, xl, yl):
    localR = labels_R[xr][yr]
    localL = labels[xl][yl]
    if labels_R[xr, yr] ==0 or labels[xl, yl] ==0:
        print("Punto no encontrado")
        return (0, 0)

    PrjPoints_R.append(centroids_R[labels_R[xr][yr]])
    PrjPoints_L.append(centroids[labels[xl][yl]])

    for a in range(0, 40):
        marca = 0
        if centroids_R[localR][0] - centroids_R[localR - 1][0] < 20 and centroids[localL][0] - centroids[localL - 1][
            0] < 20:
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
            for i in range(-7, 7):
                if marca == 1:
                    break
                for j in range(-10, 10):
                    if labels_R[xr][int(centroids_R[localR][0] - (15+i))] != 0 and labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                            labels[xl][int(centroids[localL][0] - (15+i))] != 0 and labels[int(centroids[localL][1] - j)][yl] != 0:

                        PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15+i))]])
                        PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] - (15+i))]])
                        marca = 1
                        print(labels_R[xr][int(centroids_R[localR][0] - (15 + i))])
                        print("patra", labels[xl][int(centroids[localL][0] - (15 + i))])
                        cv2.circle(labeled_img_R, (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15+i))]][0]),
                                                   int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15+i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][0]),
                                    int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][0]),
                                    int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[xr][int(centroids_R[localR][0] - (15+i))]
                        localL = labels[xl][int(centroids[localL][0] - (15+i))]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])
                        break
def Matchear_pts_Abajo(xr, yr, xl, yl):
    localR = labels_R[xr][yr]
    localL = labels[xl][yl]
    if labels_R[xr, yr] ==0 or labels[xl, yl] ==0:
        print("Punto no encontrado")
        return (0, 0)

    PrjPoints_R.append(centroids_R[labels_R[xr][yr]])
    PrjPoints_L.append(centroids[labels[xl][yl]])

    # Busca ptos consecutivos hacia abajo en vertical sumando alrededor de 20 pixeles y +-5 en horizontal
    for a in range(0, 2):
        marca = 0
        for i in range(-5, 5):
            if marca == 1:
                break
            for j in range(-5, 5):
                if labels_R[int(centroids_R[localR][1] + (15 + i))][yr] != 0 and \
                        labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                        labels[int(centroids[localL][1] + (15 + i))][yl] != 0 and \
                        labels[xl][int(centroids[localL][0] + j)] != 0:
                    PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]])
                    PrjPoints_L.append(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]])
                    marca = 1

                    print("abajo", labels[int(centroids[localL][1] + (15 + i))][yl])
                    cv2.circle(labeled_img_R,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(labeled_img,
                               (int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(imgR,
                               (int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][0]),
                                int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][1])),
                               1, (255, 255, 255), 2)
                    cv2.circle(img1,
                               (int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][0]),
                                int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][1])),
                               1, (255, 255, 255), 2)

                    localR = labels_R[int(centroids_R[localR][1] + (15 + i))][yr]
                    localL = labels[int(centroids[localL][1] + (15 + i))][yl]

                    xr = int(centroids_R[localR][1])
                    yr = int(centroids_R[localR][0])
                    xl = int(centroids[localL][1])
                    yl = int(centroids[localL][0])

                    break
def Matchear_pts_All_2(xr, yr, xl, yl, b, d, f, u):
    cv2.circle(labeled_img_R, (int(yr), int(xr)), 1, (0, 0, 255),
               2)  # las coordenadas van como me las entrega Obtener_Coord
    cv2.circle(labeled_img, (int(yl), int(xl)), 1, (0, 0, 255), 2)
    cv2.circle(imgR, (int(yr), int(xr)), 1, (0, 0, 255), 2)  # las coordenadas van como me las entrega Obtener_Coord
    cv2.circle(img1, (int(yl), int(xl)), 1, (0, 0, 255), 2)

    if labels_R[xr, yr] == 0 or labels[xl, yl] == 0:
        print("Punto no encontrado")
        return (0, 0)

    PrjPoints_R.append(centroids_R[labels_R[xr][yr]])
    PrjPoints_L.append(centroids[labels[xl][yl]])

    localR = labels_R[xr][yr]
    localL = labels[xl][yl]
    localR2 = labels_R[xr][yr]
    localL2 = labels[xl][yl]

    # Busca ptos consecutivos hacia abajo en vertical sumando alrededor de 20 pixeles y +-5 en horizontal
    def Abajo(localR, localL, xr, yr, xl, yl, d = d):

        for a in range(0, d):
            marca = 0
            for i in range(-5, 5):
                if marca == 1:
                    break
                for j in range(-5, 5):
                    if labels_R[int(centroids_R[localR][1] + (15 + i))][yr] != 0 and \
                            labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                            labels[int(centroids[localL][1] + (15 + i))][yl] != 0 and \
                            labels[xl][int(centroids[localL][0] + j)] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]])
                        PrjPoints_L.append(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]])
                        marca = 1


                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][0]),
                                    int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][0]),
                                    int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][0]),
                                    int(centroids_R[labels_R[int(centroids_R[localR][1] + (15 + i))][yr]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][0]),
                                    int(centroids[labels[int(centroids[localL][1] + (15 + i))][yl]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[int(centroids_R[localR][1] + (15 + i))][yr]
                        localL = labels[int(centroids[localL][1] + (15 + i))][yl]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])

                        break

    # Busca ptos consecutivos hacia arriba en vertical restando alrededor de 8 pixeles y +-2 en horizontal
    def Arriba(localR, localL, xr, yr, xl, yl, u = u):
        for a in range(0, u):
            marca = 0
            for i in range(-5, 5):
                if marca == 1:
                    break
                for j in range(-5, 5):
                    if labels_R[int(centroids_R[localR][1] - (15 + i))][yr] != 0 and \
                            labels_R[xr][int(centroids_R[localR][0] + j)] != 0 and \
                            labels[int(centroids[localL][1] - (15 + i))][yl] != 0 and \
                            labels[xl][int(centroids[localL][0] + j)] != 0:
                        PrjPoints_R.append(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]])
                        PrjPoints_L.append(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]])
                        marca = 1


                        cv2.circle(labeled_img_R,
                                   (int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][0]),
                                    int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(labeled_img,
                                   (int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][0]),
                                    int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(imgR,
                                   (int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][0]),
                                    int(centroids_R[labels_R[int(centroids_R[localR][1] - (15 + i))][yr]][1])),
                                   1, (255, 255, 255), 2)
                        cv2.circle(img1,
                                   (int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][0]),
                                    int(centroids[labels[int(centroids[localL][1] - (15 + i))][yl]][1])),
                                   1, (255, 255, 255), 2)

                        localR = labels_R[int(centroids_R[localR][1] - (15 + i))][yr]
                        localL = labels[int(centroids[localL][1] - (15 + i))][yl]

                        xr = int(centroids_R[localR][1])
                        yr = int(centroids_R[localR][0])
                        xl = int(centroids[localL][1])
                        yl = int(centroids[localL][0])

                        break

    # Busca ptos consecutivos hacia atras en horizontal restando alrededor de 15 pixeles y +-4 en vertical
    for a in range(0, b):
        marca = 0

        for i in range(-7, 7):
            if marca == 1:
                break
            for j in range(-10, 10):
                if labels_R[xr][int(centroids_R[localR][0] - (15 + i))] != 0 and \
                        labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                        labels[xl][int(centroids[localL][0] - (15 + i))] != 0 and \
                        labels[int(centroids[localL][1] - j)][yl] != 0:
                    PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]])
                    PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]])

                    Abajo(localR, localL, xr, yr, xl, yl, d=d)
                    Arriba(localR, localL, xr, yr, xl, yl, u=u)

                    marca = 1

                    cv2.circle(labeled_img_R,
                                (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][0]),
                                int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][1])),
                                1, (255, 255, 255), 2)
                    cv2.circle(labeled_img,
                                (int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][0]),
                                int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][1])),
                                1, (255, 255, 255), 2)
                    cv2.circle(imgR,
                                (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][0]),
                                int(centroids_R[labels_R[xr][int(centroids_R[localR][0] - (15 + i))]][1])),
                                1, (255, 255, 255), 2)
                    cv2.circle(img1,
                                (int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][0]),
                                int(centroids[labels[xl][int(centroids[localL][0] - (15 + i))]][1])),
                                1, (255, 255, 255), 2)

                    localR = labels_R[xr][int(centroids_R[localR][0] - (15 + i))]
                    localL = labels[xl][int(centroids[localL][0] - (15 + i))]

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

    # Busca ptos consecutivos hacia adelante en horizontal sumando alrededor de 8 pixeles y +-4 en vertical
    for a in range(0, f):
        marca = 0

        for i in range(-3, 3):
            if marca == 1:
                break
            for j in range(-4, 4):
                if labels_R[xr][int(centroids_R[localR][0] + (15 + i))] != 0 and \
                        labels_R[int(centroids_R[localR][1] - j)][yr] != 0 and \
                        labels[xl][int(centroids[localL][0] + (15 + i))] != 0 and \
                        labels[int(centroids[localL][1] - j)][yl] != 0:
                    PrjPoints_R.append(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]])
                    PrjPoints_L.append(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]])

                    Abajo(localR, localL, xr, yr, xl, yl, d=d)
                    Arriba(localR, localL, xr, yr, xl, yl, u=u)

                    marca = 1

                    cv2.circle(labeled_img_R,
                                (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][0]),
                                int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][1])),
                                1, (255, 255, 255), 2)
                    cv2.circle(labeled_img,
                                (int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][0]),
                                int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][1])),
                                1, (255, 255, 255), 2)
                    cv2.circle(imgR,
                                (int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][0]),
                                int(centroids_R[labels_R[xr][int(centroids_R[localR][0] + (15 + i))]][1])),
                                1, (255, 255, 255), 2)
                    cv2.circle(img1,
                                (int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][0]),
                                int(centroids[labels[xl][int(centroids[localL][0] + (15 + i))]][1])),
                                1, (255, 255, 255), 2)

                    localR = labels_R[xr][int(centroids_R[localR][0] + (15 + i))]
                    localL = labels[xl][int(centroids[localL][0] + (15 + i))]

                    xr = int(centroids_R[localR][1])
                    yr = int(centroids_R[localR][0])
                    xl = int(centroids[localL][1])
                    yl = int(centroids[localL][0])

                    break


Matchear_pts_All_2(xrp,yrp,xlp,ylp, 5, 10, 5, 10)
Matchear_pts_All_2(xrt,yrt,xlt,ylt, 100, 100, 100, 100)
Matchear_pts_All_2(xrh,yrh,xlh,ylh, 100, 100, 100, 100)
Matchear_pts_All_2(xrw,yrw,xlw,ylw, 100, 100, 100, 100)

#Matchear_pts_All_2(xrp20,yrp20,xlp20,ylp20, 5, 10, 5, 10)
#Matchear_pts_All_2(xrp21,yrp21,xlp21,ylp21, 5, 10, 5, 10)
##Matchear_pts_All_2(xrp22,yrp22,xlp22,ylp22, 5, 10, 5, 10)

Matchear_pts_All_2(xrw1,yrw1,xlw1,ylw1, 30, 30, 30, 30)
Matchear_pts_All_2(xrw2,yrw2,xlw2,ylw2, 30, 30, 30, 30)
Matchear_pts_All_2(xrw3,yrw3,xlw3,ylw3, 30, 30, 30, 30)

Matchear_pts_All_2(xrw4,yrw4,xlw4,ylw4, 20, 20, 20, 20)

#cv2.imwrite('CircleR_L.png', labeled_img_R)
#cv2.imwrite('CircleL_L.png', labeled_img)
#cv2.imwrite('CircleR.png', imgR)
#cv2.imwrite('CircleL.png', img1)

#preparar ptos para undisorted
pts_Left = np.array([(PrjPoints_L[0][0], PrjPoints_L[0][1])])
pts_Right = np.array([(PrjPoints_R[0][0], PrjPoints_R[0][1])])

for a in range(1, PrjPoints_R.__len__()):
    pto = PrjPoints_R[a][0]
    pto1 = PrjPoints_R[a][1]
    pts_Conca = np.array([(pto, pto1)])
    pts_Right = np.concatenate((pts_Right, pts_Conca))

for a in range(1, PrjPoints_L.__len__()):
    pto = PrjPoints_L[a][0]
    pto1 = PrjPoints_L[a][1]
    pts_Conca = np.array([(pto, pto1)])
    pts_Left = np.concatenate((pts_Left, pts_Conca))


pts_Left = np.float32(pts_Left[:, np.newaxis, :])
pts_Right = np.float32(pts_Right[:, np.newaxis, :])

pts_und_Left = cv2.undistortPoints(pts_Left, cameraMatrix1, distCoeffs1, R=R1, P=P1 )
pts_und_Right = cv2.undistortPoints(pts_Right, cameraMatrix2, distCoeffs2, R=R2, P=P2)

pt1 = np.reshape(pts_und_Left,(1,PrjPoints_R.__len__(),2))
pt2 = np.reshape(pts_und_Right,(1,PrjPoints_R.__len__(),2))
newPoints1, newPoints2 = cv2.correctMatches(F, pt1, pt2)

Tpoints1 = cv2.triangulatePoints(P1, P2, newPoints1, newPoints2)
#Tpoints1 = cv2.triangulatePoints(P1, P2, pt1, pt2)

# Calculate the final time.
elapsed_triangulate = time() - start_time
print("Triangulate time: %0.10f seconds." % elapsed_triangulate)
print(newPoints1[0][0])
print(P1[0])
# Creamos la figura
fig = plt.figure()
fig2 = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')


x = Tpoints1[0]/Tpoints1[3]
y = Tpoints1[1]/Tpoints1[3]
z = Tpoints1[2]/Tpoints1[3]

#Rotacion de ejes
#Ver.....https://www.youtube.com/watch?v=qmuM_2yL90I
#Angulo de rotacion sobre el eje y
q = -0.054
m = []
for a in range(0, len(x)):
    y[a] = y[a]
    x[a] = x[a]*math.cos(q) + z[a]*math.sin(q)
    z[a] = -x[a]*math.sin(q) + z[a]*math.cos(q)
    z[a] = -(z[a] - 330)
    m.append(abs(z[a]*100/15))

ax1.scatter(x, y, z, c=m)
ax1.set_title("Triangulación")
ax1.set_xlabel('Eje x cm')
ax1.set_ylabel('Eje y cm')
ax1.set_zlabel('Eje z cm')
ax1.set_zlim(0, 100)

cv2.waitKey()


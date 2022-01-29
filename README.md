# 3D-Reconstruction-of-metal-sheet
Is a method for pairing the same points between stereoscopic images to reconstruct and estimate superficial deformation of metal sheets in mechanical stamping processes. For the procedure the metal sheets are engraved with a uniform pattern of circles before being deformed; Therefore, when measuring the initial and final 3D positions of the recorded points, sufficient information is obtained to calculate the superficial deformation.

![image](https://user-images.githubusercontent.com/87040483/151662667-e98a7836-0d10-4053-a1c0-a8011b0e6c28.png)
![image](https://user-images.githubusercontent.com/87040483/151662678-d3cb38bf-f2dd-47c0-9ad5-e62a1beb0839.png)

## Method

A pair of Prosilica GT2750 cameras were used. These are 6.1 megapixels each, with a Gigabit Ethernet port compatible with GigE Vision and a Hirose I / O port. This kind of camera incorporates a high-quality Sony sensor ICX694 EXview HAD CCD that provides extra sensitivity, near IR response, low noise level, anti-blooming effect and excellent image quality. At full resolution, this camera processes 19.8 frames per second. With a smaller region of interest, higher frame rates are possible. It is a robust device designed to operate in extreme environments and fluctuating lighting condi-tions. It offers precise control of the iris lens that allows users to set the size of the aperture to optimize depth of field, exposure and gain without the need for additional control elements. For the assembly, a metal structure of 40x40x40cm was designed, the piece to be measured is approximately at 30 cm from the lens of the cameras. LED lighting was chosen due to the contrast with the circles marked on the sheet: blue LED of 640 nm. The acquired images have a resolution of 2752x2200 pixels.

The followed methodology has the purpose of measuring the surface deformation in a metal sheet used for car bodies. 
1.	Calibrations of the cameras.
2.	Stamping of known circle grid on the not deformed metallic sheet.
3.	Deformation of the metal-sheet by the mechanical stamping process.
4.	Illumination of the piece with LED blue light for being measured.
5.	Capture of stereo images.
6.	Digital image processing to obtain the labels of the points of the metal sheet.
7.	Manual selection of the same point in both images to allow the algorithm to matching all the remaining points among them.
8.	Triangulation of points to obtain their position in 3D space and reconstruction of the metal sheet.
9.	Estimate the deformation by two methods: 
   •	From the average distances of each point with its neighbors in 3D space.
   •	Through the depth value of each point (measurement on Z axis).

For the individual and stereo calibration of the cameras, the defined functions provided by the OpenCV library were used within the Python programming language

![image](https://user-images.githubusercontent.com/87040483/151662604-ada98399-39b5-4bb5-b671-ae952be4205b.png)

![image](https://user-images.githubusercontent.com/87040483/151663913-ff5c39c9-8e11-49af-992a-42b941b74bbc.png)


### 	Digital image processing 

![image](https://user-images.githubusercontent.com/87040483/151663596-cdc1426e-df86-4103-85e3-81acccf25865.png)

### Neighbor search algorithm

The method’s aim is to determine the correspondences between 2D centroid coordinates of both images for each circle of the grid pattern on the metal piece giving only one click on each image (left and right). First a click is given on any circle on the image taken by a left camera, then another click is given at that same circle but on the other image taken by the right camera. With this, the centroid of the labels corresponding to these marks is determined and the first correspondence between both images is obtained manually (red circle in figure). Then a search of neighboring labels in areas near the central circle is made, here are observed in yellow boxes the regions where tags are searched in both images. A new correspondence is obtained if both labels are found in the corresponding boxes of both images. The process is iterative, taking as a central point each new matched label correctly.

![image](https://user-images.githubusercontent.com/87040483/151663740-8a7bcaf4-01fd-413b-9d20-9458e35da7a6.png)


 ##  Results
 
 ![image](https://user-images.githubusercontent.com/87040483/151663773-b2c4cff1-f626-46a7-9a4e-baa648223f25.png)

 
 

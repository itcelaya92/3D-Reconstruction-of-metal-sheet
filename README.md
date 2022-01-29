# 3D-Reconstruction-of-metal-sheet
Is a method for pairing the same points between stereoscopic images to reconstruct and estimate superficial deformation of metal sheets in mechanical stamping processes. For the procedure the metal sheets are engraved with a uniform pattern of circles before being deformed; Therefore, when measuring the initial and final 3D positions of the recorded points, sufficient information is obtained to calculate the superficial deformation.

![image](https://user-images.githubusercontent.com/87040483/151662667-e98a7836-0d10-4053-a1c0-a8011b0e6c28.png)
![image](https://user-images.githubusercontent.com/87040483/151662678-d3cb38bf-f2dd-47c0-9ad5-e62a1beb0839.png)

## Method

![image](https://user-images.githubusercontent.com/87040483/151662604-ada98399-39b5-4bb5-b671-ae952be4205b.png)

In the measurement system is indicated by the user, the position of two neighboring marks in each image to have an initial distance and slope of search of their centroids. From the corresponding centroids the stereoscopic triangulation is computed and with this the position of the centroids in 3D and finally the deformation is calculated by averaging the differences of the distances with respect to the reference distance with its four neighbors divided by the reference distance.


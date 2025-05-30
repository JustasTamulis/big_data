# Data mining

Use the env from lab1

# Lab 5

The goal of the assignment is to detect marine transportation ports in the dataset. (All sub-tasks shall be done in a parallel manner (python parallel, pyspark or etc.))

Free to choose reasonable and creative ways to define port parameters, but please avoid over-engineering. The most complex solution doesn't mean the best. 


## Filter out the noise and prepare data for port detection. (Think what is the noise in this solution)

Ships only
Remove the sudden jump points, not the whole MMSI, but the points only.

## Create an algorithm for port detection.  

Moving slow,
Clustering, but without time?
Cluster the locations only when no movement or very slow speeds ?
Must have many different MMSI clustered

## Evaluate the relative size of the port.

From the clustered coordinates - get max/min x, y. Calculate grid size.


## Visualize ports. (Get creative with it.) It's good to know the location of the port, its relative size, etc.

Use folium to place a mesh on clusters. Include their size, number of points and unique MMSI count.


## Solution presentation. Maximum of 5 slides presenting solutions.

Latex in overleaf. Slides: 1) Overview 2) Filtering 3) Port detection 4) Port size calculations 5) Resulting map
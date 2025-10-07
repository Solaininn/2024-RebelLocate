# 2024-RebelLocate
A machine learning program designed to guess the location of images using coordinate data and Python distance algorithms, CNN, and KNN.

The program utilizes a self-sourced image database containing over 6,000 individual images across the UNLV college campus. This database was created in collaboration with other students at UNLV as a group project. Each picture contains EXIF data, specifically location coordinates, which were used in the tuning of the distance algorithm. 

The multiple files are to distribute each step into different Python files. RebelLocate - Zoli.py is a collection of all the Python files used for the final result, which include CNN.py and KNearestN.py. The CoordExtract files are for taking our Excel sheet of location data, that being buildings.csv, and using KNN to gauge the distance between each image and the building locations, sorting every image based upon the building it is closest too, outputing another excel sheet in data.csv.

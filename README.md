# NAME

    3D Laser Sintering Optimisation

[Dissertation PDF](https://github.com/ThomasAstley/3D_Laser_Sintering_Optimisation/blob/main/Dissertation.pdf)

![result](https://github.com/ThomasAstley/3D_Laser_Sintering_Optimisation/blob/main/result_images/parts.png)

# DESCRIPTION

Use machine learning to optimise Laser Powder Bed Fusion process parameters for dimensional tolerance

New machines are built with different sets of adjustable parameters and new metal alloys are produced with different properties, resulting in different parameters being required to previous alloys and machines. These need to be optimised before usable components can be printed.

In this dissertation, machine learning combined with image analysis is used to obtain predictions for optimum values for two parameters of the laser powder bed fusion process, beam compensation (BC) and contour distance (CD); the main impact of these parameters is on the dimensional tolerance and accuracy of the smaller sections of the resulting product.

# BC AND CD OPTIMISATION 

## Using Computer Vision For Thickness Measurement

Canny edge detection was used in order to detect the edges of a section of a test product, and then an average width was determined from this. Then this was used to calculate a percentage error based on the target width for that section.

![result](https://github.com/ThomasAstley/3D_Laser_Sintering_Optimisation/blob/main/result_images/canny_edge_detection.png)

## Deriving The Thickness To BC/CD Equation

The results from the thickness measurement are fed into a gradient decent algorithm, along with the BC and CD values to produce an equation that relates them to each other. 

## Optimising The Thickness To BC/CD Equation

Simulated annealing is then used to find the minimum of the equation, in the bounds of the minimum and maximum allowed BC and CD values. 

![result](https://github.com/ThomasAstley/3D_Laser_Sintering_Optimisation/blob/main/result_images/gradient_decent_graph.png)

# RESULTS

The computer vision for thickness measurement is a significant improvement over human measurement.

The BC/CD parameters produced worked well for each thickness except for 0.2mm. 

The beam compensation (BC) has little to no effect on the accuracy of the thickness, except at 0.2mm thickness.

At 0.2mm thickness BC plays a significant role, which may explain why the predicted overall optimal BC and CD did not work as well for that thickness. 

# COPYRIGHT AND LICENSE  

Thomas Astley 2023

GNU GPL Version 3, see *LICENSE.txt* file


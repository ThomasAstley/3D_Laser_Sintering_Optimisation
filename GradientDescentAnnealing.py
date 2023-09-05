import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import time
import Vision

# This while loop is used to ensure that the user enters a valid input, and then runs the vision analysis if the user chooses to.
while True:
    try:
        remeasure = int(input('Press 1 to run vision analysis to obtain new measurements, or 2 to skip: '))
        if remeasure in range(1, 3):
            break
        else:
             raise ValueError
    except ValueError:
        print('Invalid input. Please try again.')
if remeasure == 1:
    Vision.main()        

#This while loop is used to ensure that the user enters a valid input, and then reads the data from the csv file.
while True:
    try:
        measurements = int(input('Press 1 to use the initial measurements or 2 to use measurements obtained by vision analysis: '))
        if measurements in range(1, 3):
            break
        else:
            raise ValueError
    except ValueError:
        print('Invalid input. Please try again.')
        

# Read data from csv file
data=[]
with open('Resources/Book'+str(measurements)+'.csv', encoding='UTF-8-sig') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)
    csv_file.close()

# store data from each column in a list and remove the first row (headers).
x1=[row[0] for row in data]
x2=[row[1] for row in data]
x1.pop(0)
x2.pop(0)

# This while loop is used to ensure that the user enters a valid input, and then reads the data from the csv file.
while True:
    try:
        width = int(input('Enter the width that you want to obtain optimum parameters for (1 for 1mm | 2 for 0.7mm | 3 for 0.5mm | 4 for 0.4mm | 5 for 0.3mm | 6 for 0.2mm | 7 for all): '))
        if width in range(1, 8):
            break
        else:
            raise ValueError
    except ValueError:
        print('Invalid input. Please try again.')
# calculate percent error from target for each value, remove first row (headers) and then store in a list.
if width in range(1,7):
    y=[row[width+1] for row in data]
    for i in range(1, len(y)):
        y[i]=abs((y[i]-y[0])/y[0])*100
    y.pop(0)


# add x1 and x2 to themselves 6 times as all values was selected as input, as there are 6 y values for each x1 and x2 value, 
# but we need 1 y value for each x1 and x2 value.
else:
    x1 = x1+x1+x1+x1+x1+x1
    x2 = x2+x2+x2+x2+x2+x2
    y = []
    for i in range(2, 8):
        y_temp = [row[i] for row in data]
        for j in range(1, len(y_temp)):
            y_temp[j]=abs((y_temp[j]-y_temp[0])/y_temp[0])*100
        y_temp.pop(0)
        y=y+y_temp

#stores the original values for x1, x2 and y, used to plot the data and the graph of the equation later.
x1_original = x1
x2_original = x2
y_original = y


#feature scaling for x1, x2 and y. Used to speed up convergence of the gradient descent algorithm.
mean_x1 = sum(x1) / len(x1)
mean_x2 = sum(x2) / len(x2)
mean_y = sum(y) / len(y)
std_x1 = math.sqrt(sum((x - mean_x1)**2 for x in x1) / len(x1))
std_x2 = math.sqrt(sum((x - mean_x2)**2 for x in x2) / len(x2))
std_y = math.sqrt(sum((x - mean_y)**2 for x in y) / len(y))
x1 = [(x - mean_x1) / std_x1 for x in x1]
x2 = [(x - mean_x2) / std_x2 for x in x2]
y = [(x - mean_y) / std_y for x in y]
start_time = time.time()

# Gradient descent algorithm
# a is the learning rate, b is the momentum, c1 is the error from the previous iteration, c2 is the error from the current iteration,
# E is a small number used to prevent division by 0.
a=0.01
b=0.9
c1=0
c2=1
E = 1e-8
iterations = 0
length = len(y)
x1_squared = [x1[i]**2 for i in range(0, length)]
x2_squared = [x2[i]**2 for i in range(0, length)]
x1_x2 = [x1[i]*x2[i] for i in range(0, length)]
print('Starting iterations for gradient descent...')
w0, w1, w2, w3, w4, w5 = 0, 0, 0, 0, 0, 0
g0, g1, g2, g3, g4, g5 = 0, 0, 0, 0, 0, 0
v0, v1, v2, v3, v4, v5 = 0, 0, 0, 0, 0, 0

# Loop until the change in error (C) is less than 0.0000001.
# The algorithm is attempting to find values for w0 - w5 for the equation y = w0 + w1*x1 + w2*x2 + w3*x1^2 + w4*x1*x2 + w5*x2^2, 
# such that the difference between the values and the graph is minimized.
# The algorithm uses momentum to try to avoid getting stuck in a local minimum and Adagrad to adjust the learning rate for each weight.
while abs(c2-c1) > 0.000001 and iterations < 100000:
    c=0
    for i in range(0, length):
        w1x, w2x, w3x, w4x, w5x = x1[i], x2[i], x1_squared[i], x1_x2[i], x2_squared[i]
        f = w0 + w1*w1x + w2*w2x + w3*w3x + w4*w4x + w5*w5x
        difference = f - y[i]
        d1, d2, d3, d4, d5 = difference*w1x, difference*w2x, difference*w3x, difference*w4x, difference*w5x
        g0 = g0 + difference**2
        g1 = g1 + (d1)**2
        g2 = g2 + (d2)**2
        g3 = g3 + (d3)**2
        g4 = g4 + (d4)**2
        g5 = g5 + (d5)**2
        v0= b*v0+ (a/(math.sqrt(g0+E)))*difference
        v1= b*v1+ (a/(math.sqrt(g1+E)))*d1
        v2= b*v2+ (a/(math.sqrt(g2+E)))*d2
        v3= b*v3+ (a/(math.sqrt(g3+E)))*d3
        v4= b*v4+ (a/(math.sqrt(g4+E)))*d4
        v5= b*v5+ (a/(math.sqrt(g5+E)))*d5
        c = c + difference**2
        w0 = w0 - v0
        w1 = w1 - v1
        w2 = w2 - v2
        w3 = w3 - v3
        w4 = w4 - v4
        w5 = w5 - v5
        
    c = c / (2*length)
        
    iterations += 1
    c2 = c1
    c1 = c
    
           
print('Finished in {:.2f} seconds!'.format(time.time() - start_time), 'With {} iterations'.format(iterations))
print('Error: {:.4f} | w0: {:.4f} | w1: {:.4f} | w2: {:.4f} | w3: {:.6f} | w4: {:.6f} | w5: {:.5f}'.format(c, w0, w1, w2, w3, w4, w5))

#function to calculate the error
def f(x1, x2): return w0 + w1*x1 + w2*x2 + w3*(x1**2) + w4*(x1*x2) + w5*(x2**2)

# Simulated Annealing algorithm
current_x1 = np.random.uniform(min(x1), max(x1))
current_x2 = np.random.uniform(min(x2), max(x2))
optimum_x1 = current_x1
optimum_x2 = current_x2
temperature = 1000.0
cooling_rate = 0.95
iterations_per_temp = 100
lowest_error = f(optimum_x1, optimum_x2)
start_time = time.time()
print('Starting iterations for simulated annealing...')
# Loop until the temperature is less than 0.1. 
while temperature > 10:
    for iteration in range(iterations_per_temp):
        new_x1 = np.clip(current_x1 + np.random.normal(scale=1), min(x1), max(x1))
        new_x2 = np.clip(current_x2 + np.random.normal(scale=1), min(x2), max(x2))
        current_error = f(current_x1, current_x2)
        new_error = f(new_x1, new_x2)
        # If the new error is less than the current error or a random number between 0 and 1 is less than e^((current error - new error)/temperature),
        # it updates the values for the current x1 and x2, the likelihood of taking a "bad" value happening decreases as the temperature decreases.
        #This is done to try to avoid getting stuck in a local minimum.
        if new_error < current_error or np.random.rand() < np.exp((current_error - new_error) / temperature):
            current_x1 = new_x1
            current_x2 = new_x2
            current_error = new_error
            if new_error < lowest_error:
                optimum_x1 = new_x1
                optimum_x2 = new_x2
                lowest_error = new_error
    temperature = temperature * cooling_rate
    
optimum_x1 = optimum_x1 * std_x1 + mean_x1
optimum_x2 = optimum_x2 * std_x2 + mean_x2
lowest_error = lowest_error * std_y + mean_y
print('Finished in {:.2f} seconds!'.format(time.time() - start_time))
print('Optimum BC: {:.4f} | Optimum CD: {:.4f} | Lowest error: {:.4f}'.format(optimum_x1, optimum_x2, lowest_error))
#using the weights from gradient descent, this takes the unscaled x1 and x2 values and calculates the y value.
def f_unscaled(x1, x2): 
    return (w0 + w1*((x1 - mean_x1) / std_x1) + w2*((x2 - mean_x2) / std_x2) + w3*(((x1 - mean_x1) / std_x1)**2) + 
            w4*(((x1 - mean_x1) / std_x1)*((x2 - mean_x2) / std_x2)) + w5*(((x2 - mean_x2) / std_x2)**2))*std_y + mean_y
# Plot the data and the graph of the equation.
# The graph of the equation is a 3D graph, so the data is plotted as a scatter plot and the graph is plotted as a surface plot.
ax = plt.axes(projection='3d')
ax.scatter(x1_original, x2_original, y_original)
X1 = np.linspace(min(x1_original), max(x1_original), 100)
X2 = np.linspace(min(x2_original), max(x2_original), 100)
X11, X22 = np.meshgrid(X1, X2)
ax.plot_surface(X11, X22, f_unscaled(X11, X22), cmap='viridis')
ax.set_xlabel('BC')
ax.set_ylabel('CD')
ax.set_zlabel('Percent Error')
plt.show()
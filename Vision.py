import cv2
import numpy as np
import csv
import time

    # Function to perform edge detection on an image
def edge_detection(image, low_threshold, high_threshold):
    # Gaussian filter is applied in order to smooth the image. Goal is to suppress noice without destroying the edges.
    gaussian_blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detection from opencv to detect edges. Pixels with a gradient larger than the high_threshold are considered sure to be edges.
    # Pixels with a gradient smaller than the low_threshold are considered to be not edges.
    # Pixels with a gradient between the two thresholds are considered to be edges only if they are connected to pixels that are sure to be edges.
    canny_edges = cv2.Canny(gaussian_blurred_image, low_threshold, high_threshold)
    return canny_edges


    #function to check if values appear twice in a list. Used later to check if both sides of the wall are detected.
def check_values_appear_twice(list):
    count = {} 
    for i in list:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    for i, count in count.items():
        if count < 2:
            return False

    return True


    # Function to count the number of pixels between the first and last blue pixel in a line for each row of pixels in the image.
    # Used to calculate the area of the wall in terms of pixels.
def count_pixels(blue_pixel_positions):

    # Seperates the x and y coordinates of the blue pixels into seperate lists.
    blue_x_coordinates = blue_pixel_positions[1]
    blue_y_coordinates = blue_pixel_positions[0]
    
    # Creates a dictionary of the lowest and highest x coordinate for each row of pixels in the image.
    blue_pixel_coordinates_dict = {}
    for x, y in zip(blue_x_coordinates, blue_y_coordinates):
        if y not in blue_pixel_coordinates_dict:
            blue_pixel_coordinates_dict[y] = {'lowest_x': x, 'highest_x': x}
        else:
            if x < blue_pixel_coordinates_dict[y]['lowest_x']:
                blue_pixel_coordinates_dict[y]['lowest_x'] = x
            elif x > blue_pixel_coordinates_dict[y]['highest_x']:
                blue_pixel_coordinates_dict[y]['highest_x'] = x
    
    # Calculates the total number of pixels between the first and last blue pixel in each row of the image and adds them together.
    total = 0
    for y, x in blue_pixel_coordinates_dict.items():
        total = total + (x['highest_x'] - x['lowest_x'])
            
    return total

    # Function to calculate the average thickness of the wall in mm for the images. 
    # Different reference image used for first 13, because the final 12 images were taken with different equipment and have a different scale.
reference_image1 = cv2.imread('Images/#1/w1.jpg', cv2.IMREAD_GRAYSCALE)
length_bar1 = reference_image1[969:970, 300:1280]
length_pixels1 = np.sum(length_bar1 < 50)
PIXEL_LENGTH1 = 2/length_pixels1

reference_image2 = cv2.imread('Images/#14/w5_1.tif', cv2.IMREAD_GRAYSCALE)
length_bar2 = reference_image2[329:330, 900:1280]
length_pixels2 = np.sum(length_bar2 > 200)
PIXEL_LENGTH2 = 0.2/length_pixels2

#function to calculate the average thickness of the wall in mm for the images.
def calculate_average_thickness(image, blue_pixel_positions,i):
    area = count_pixels(blue_pixel_positions)
    height = image.shape[0]
    if i<=13:
        average_thickness = (area/height)*PIXEL_LENGTH1
    else:
        average_thickness = (area/height)*PIXEL_LENGTH2
    return average_thickness

# Initialize the results list with the collumn names.
results = [[1,1,1,0.7,0.5,0.4,0.3,0.2]]

def main():
    start_time = time.time()
    # Loop through all images to find the average width of each line. 
    print('Starting analysis...')
    for i in range(1, 26):
        
        # Initialises result row for the first 2 columns, BC and CD. Both start at 0 initially. 
        # BC increases by 25 each test, reseting to 0 every 5 tests. CD increases by 25 every 5 tests.
        result_set = [(i-1)%5*25, int((i-1)/5)*25]
        for j in range(1, 7):
            current_iteration_start_time = time.time()
            # Reads the image from the file. Different file types are used for the first 10 images.
            if i<=10:
                reference_image = cv2.imread('Images/#'+str(i)+'/w'+str(j)+'.jpg', cv2.IMREAD_GRAYSCALE)
            else:
                reference_image = cv2.imread('Images/#'+str(i)+'/w'+str(j)+'.tif', cv2.IMREAD_GRAYSCALE)
                
            # Crops the image to remove the scale bar and the bottom of the image for the first 13 images, as it is not nessesary on the last 12.
            # also crops to focus on the wall, removing most of the background, not nessesary for the last 12 images.
            if i<= 13:
                image = reference_image[0:960, 400:900]
            else:
                image = reference_image
                
            # Sets the initial values for the low and high thresholds for the edge detection.
            low_threshold = 0
            high_threshold = 100
            
            # Loops through the edge detection until the wall is detected on both sides.
            while True:
                # Performs the edge detection
                edges = edge_detection(image, low_threshold, high_threshold)

                # Creates a copy of the image in RGB format in order to display the detected edges in blue. 
                # This is useful, as the rest of the image is in grayscale, so the pixels coloured for the edges will be easily identifiable.
                image_rgb = cv2.merge((image, image, image))

                # Creates a mask of the edges
                edges_mask = edges.astype(bool)

                # Sets the overlay colour on the detected edges in the original image to blue.
                image_rgb[edges_mask] = (255, 0, 0)
                
                # finds the locations of the blue pixels in the image. 
                blue_pixel_positions = np.where(np.all(image_rgb == [255,0,0], axis=-1))

                # Checks if blue pixels appear twice in each row of the image. If they do it breaks out of the loop. 
                # If not, it retrys with a lower upper threshold.
                
                if check_values_appear_twice(blue_pixel_positions[0]):
                    break
                else:
                    high_threshold = high_threshold - 25


            # Calculates the average thickness of the wall in mm for the image.
            current_iteration_end_time = time.time()
            average_thickness = calculate_average_thickness(image_rgb, blue_pixel_positions,i)
            print(i,'|', j,'|', high_threshold, '|',average_thickness, '|', (current_iteration_end_time - current_iteration_start_time), 'seconds')
            result_set.append(average_thickness)
            #un-comment to show images while testing
            """image_rgb = cv2.resize(image_rgb, (0,0), fx=0.75, fy=0.75)
            cv2.imshow('Edge Detection Result', image_rgb)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
        # Adds the row of results to the results list.
        results.append(result_set)

    print('Done! Total time taken is ', (time.time() - start_time), 'seconds.')

    # Writes the results to a csv file.
    csv_file_path='Resources/Book2.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)
        csvfile.close()
        
# Run the main function if this file is run directly.
if __name__ == "__main__":
    main()
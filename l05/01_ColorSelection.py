import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('test.jpg')
print("image type {}".format(type(image)))
print("image.shape {}".format(image.shape))

# Grab the x and y size and make a copy of the image
xsize = image.shape[0]
ysize = image.shape[1]
color_select = np.copy(image)
print("xsize {}, ysize {}".format(xsize, ysize))

# Define color selection criteria
###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
red_threshold = 200
green_threshold = 200
blue_threshold = 200
######

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Do a boolean or with the "|" character to identify
# pixels below the thresholds
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
#Swapnil: thresholds is a 2d array of type boolean
            
print("thresholds.shape {}, {}".format(thresholds.shape, thresholds))
print("type(thresholds[0][0]) {}".format(type(thresholds[0][0])))
color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)

# Uncomment the following code if you are running the code locally and wish to save the image
# mpimg.imsave("test-after.png", color_select)

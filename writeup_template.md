## Advanced Lane Finding Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

Below is a brief writeup for the Advanced Lane Line Finding project.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first and second code cells of the IPython notebook (Advanced_Lane_finding.ipynb).

I started by preparing the "object points", which will be the x, y, and z coordinates of the chessboard corners. Since z=0, we can ignore z and focus solely on x and y. 'objp' is thus a replicated array of coordinates, which is appended to 'objpoints' with a new copy every time chessboard corners are deteced. 'imgpoints' holds x and y pixel positions of each of the corners in the image plane with every new chessboard detection. 

Then, I used 'objpoints' and 'imgpoints' to calibrate and undistort. Calibration was completed using 'cv2.calibrateCamera()', which then fed into 'cv2.undistort()' by taking the distortion coefficient from the calibration function.

Check out the image below for a quick example:

![alt text][INSERT IMAGE LINK HERE "Undistorted chessboard"]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I ran test1.jpg through the calibration and undistort function to confirm the function was working correctly. See below for an example:

![alt text][INSERT "Undistorted test1.jpg"]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color transforms and gradients to created a thresholded binary image that turned out to be very successful when building a final pipeline. 

In code cell 5 of the IPython notebook, you can check out three different gradients that I ran for testing purposes. 

The first is 'abs_sobel_thresh', which converts an image to grayscale and then takes either the x or y oriented sobel threshold and applies it to the image.

The second is 'mag_thresh', which converts to grayscale, takes sobel in the x and y direction, then uses sobelx and sobely to find a magnitude. This is finally used to breate a binary mask after being scaled to 8-bit and converted to unint8.

The final gradient is 'dir_threshold', which converts to grayscale, takes sobel in the x and y direction, creates the absolute value for both sobelx and sobely, and then finally uses np.arctan2 to calculate the direction of the gradient. This is then used to create a binary mask where the direction thresholds are met. Examples of the various gradients can be seen below:

Sobel X
![alt text][image3 "Sobel X"]

Sobel Y
![alt text][image3 "Sobel Y"]

Magnitude Threshold
![alt text][image3 "Magnitude"]

Direction Threshold
![alt text][image3 "Direction"]

Combined Threshold
![alt text][image3 "Combined"]

I then began testing various color transforms to get a sense for what would be most effective in my pipeline. I tested HLS, HSV, and LAB and was able to view each channel individually. Example images can be found by opeining up the IPython notebook in the browser and navigating to code cell 6.

Most importantly, I then created thresholds for each color space to test as well. You can view them all in code cell 7. I defined one for HLS, lightness (again utilizing HLS), HSV, LAB, and finally a combined threshold. 

Here are examples for Lightness, HLS, HSV Value, HSV Saturation, and the LAB binaries.

Lightness Binary
![alt text][image3 "Lightness"]

HLS Binary
![alt text][image3 "HLS"]

HSV Value Binary
![alt text][image3 "HSV value"]

HSV Saturation Binary
![alt text][image3 "HSV saturation"]

LAB Binary
![alt text][image3 "LAB"]

The combined binary function took quite a bit of time to work through to find the right combination of thresholds that yielded the best results. As you can see, in code cell 10 of the IPython notebook, I ended up utilizing a combination of 7 thresholds to come up with my final image processing function. An example of the final result is below:

Combined Binary
![alt text][image3 "Combined binary"]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called 'unwarp' in code cell 11 of the IPython notebook. The 'unwarp' function takes as image as input, gathers the height and width, and takes in source and destination points as well. I chose to hardcode the source and destination points like so:

```python
src = np.float32([[585,460], 
                  [203,720],
                  [1127,720], 
                  [695,460]])
dst = np.float32([[320,0], 
                  [320,720],
                  [960,720], 
                  [960,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

You can see an example of the output of the 'unwarp' function by checking out the image below:

Perspective Transform
![alt text][image4 "Perspective"]

I then ran all of the test images through the 'pipeline' function to test its performance. This allowed me to see if the color space and gradient thresholds were performing well enough for the final test with the video. You can check out the pipeline and test images in code cells 12 and 13, respectively.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

By following the instructions closely from the Lessons provided, I identified lane-line pixels and fit their positions within code cells 14 - 18. 

I began by creating a histogram based on the shape of the binary image. Since pixel values are either 0s or 1s, a peak in the histogram will indicate that there is a lane line. See the example image below:

Histogram
![alt text][image5 "Histogram"]

By following along in the Lesson, I then fit my lane lines with a 2nd order polynomial with the same code that Udacity had provided. It turned out to be plenty robust in fitting lane lines. You can find the code in cells 15 - 18 of my IPython notebook. Here are couple visualations of the lane lines being detected.

Lane Lines 1
![alt text][image5 "Lane lines 1"]

Lane Lines 2
![alt text][image5 "Lane lines 2"]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In code cells 19 - 21, I then calculated the radius of curvature and center offset. 

First, in code cell 20, I calculated a meters per pixel in both the x and y dimensions. I then fit the new polynomials to x,y in world space, calculated the new radii of the curvature, and then finally the vehicle position with respect to center. Finally, I created strings for both an necessary pieces (radii, center offset, etc.) that could be used as test overlays in my final video. 

Resources:
- [Where is the vehicle?](https://discussions.udacity.com/t/where-is-the-vehicle-in-relation-to-the-center-of-the-road/237424/2?u=tim.lapinskas)
- [Radius of curvature](https://discussions.udacity.com/t/radius-of-curvature-large-values/249332/9)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 22 of the IPython notebook. By following great instructions provided by the Udacity Lesson, I was able to successfully overlay the resulting lane area onto the original image along with the radius of curvature and center offset. An example image is below:

Lane Area and text overlay
![alt text][image6 "Lane area"]

Huge shoutout to the second resource link below. It really brought me over the hump by giving me the idea to combine many many thresholds. At first, I was only combining a couple thresholds and the lane area was very distorted during my first few videos. My final result would not be nearly as good without the second link's discussion.

Resources:
- [Lane area problems](https://discussions.udacity.com/t/the-green-surface-is-not-covering-the-lanes-properly/357275)
- [Surface distortion](https://discussions.udacity.com/t/green-surface-is-distorted/367543)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The final pipeline combines two major functions, 'pipeline' and 'lane', which are combined into a larger image processing function called 'final_pipeline'. By utilizing moviepy.editor (VideoFileClip), I was able to run each video frame through 'final_pipeline' successfully.

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I haven't had the time to test the challenge videos on my pipeline, but I did have a chance to watch them. I believe the shadows and amount of light entering each frame will cause major disruptions to my image pipeline.

I would have to mess around quite a bit more with the color space and gradients for each challenge videos before I could successfully create lane areas. I believe the extra light entering harder_challenge_video.mp4 would be quite challenging to fix. A quick Google search though, yields some quick results that could be used.

The two that I would most likely pursue first are brightness and contrast settings for each image. I would create brightness and contrast threshold functions that would specifically be used to process harder_challenge_video.mp4.

To deal with the shadows in challenge_video.mp4, I would follow the advice of [this link](https://dsp.stackexchange.com/questions/2247/how-can-i-remove-shadows-from-an-image). It seems there have been numerous studies done to remove shadows from images, which seems promising.

Finally, I believe this pipeline is not perfect when encountering various pavement colors. While, I was able to negate the effects of different colored pavement in my current pipeline, you can see that it is not perfect on the lighter colored pavement. Further investigation and standardization of pavement coloring with respect to lane lines would be very helpful.

Moving on to even tougher challenges (e.g. rain, snow, nighttime, etc.) would most likely prove to be even tougher. An immense about of research would have to be done within each of these domains and would most likely take a few months for each subject. Thinking about the varying conditions, you can start to see why image processing using only camera images is so difficult. Recently, Keras create Francis Chollet tweeted about these difficulties in comparision to training neural networks. The amount of time needed to fine tune image processing versus neural networks is so large that even the highest performing teams will take years to create a self-driving car solely with image data. 

I would be very interested to see how many of the large firms are combining lidar, radar, camera data, and neural networks to solve these problems. Obviously the more data, the more robust the models and pipelines can become and I would love to explore this sometime in the future. 

I will also be checking out the followiing dicussions to further my knowledge in this topic. The forum has turned out to be my most useful resource for these projects and extending past the projects we are completing.

Resources:
- [Mathematical application for distortion](https://discussions.udacity.com/t/any-good-resources-for-the-mathematical-parts-of-distortion/330382)
- [Industry standards for lane-finding](https://discussions.udacity.com/t/does-anyone-know-what-kinds-of-the-lane-finding-methods-are-used-in-autonomous-driving-companies/249145/2)

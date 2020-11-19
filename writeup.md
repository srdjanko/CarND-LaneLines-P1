**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

---

### Reflection

## Pipeline consists of the several steps

**Converting images to grayscale**

We only need brightness gradient information from the image, and grayscale contains that information.

**Gaussian blur**

Bluring the image removes some amount of noise from the image. This makes it easier to detect actual edges in the image.

**Canny edge detection**

Detect edges on the image. Simply explained, algorithm calculates image gradient (ie using the Laplacian), and then forms the edges in two steps: detect strong edges first then extended them following the weaker edges in between.

**Hough transform**

After the edges have been detected, we need to detect what *could* be straight lines among those edges. As part of this step we keep only the part of the image that is of interest for the lane detection, and simply disregard the rest. The said part of the image corresponds to the area of the road where we roughly expect to find road lines (which in this approximation are also assumed to be straight lines).

Easiest way for the computer to detect straigh lines is to first map all pixels from the image space to the Hough space. Ideally in the Hough space all pixels/points that are part of the same line will map to the same point, and those that are *close* to be on the same line will be near each other in the Hough space. So, the algorithm counts the number of points in the each *cell* of the Hough space, to establish the existance of lines connecting these pixels. Size of the *cell* will depend on the theta and rho resolution parameters.

**Interpolate detected lines to form lane lines**

When we have detected all straight lines in the image, we can *average* them in some way to form single line for each lane. In this reworked solution I have used some of the suggestions received when I submited the project.

For interpolating detected segments to form lane lines, I have used simple averaging algorithm where:

1. Lane direction/slope is calculated by averaging all the segment projections on image axis.
2. Containing point on the line was calculated as the weighted sum of the middle point of projections, where the *weights* are segment lengths. This works just exactly as determining the *centre of mass* for all the line segments.

I also used *averaging with priors* as suggested in the review, which helped to stabilize the detections quite significantly.

## Potential shortcomings with current pipeline

For detected line segments that are further away, perhaps it would make sense to assign weights to compensate the fact they must be visualy smaller. Similarly, the brightness of the road parts that are further away is expectedly lesser. These effects are not compensated.

For the *challenge* assignment, most difficult part was to obtain decent Hough lines from the video. There is a part of the video where the detection fails almost completely.

## Possible improvements

Perhaps some transformation that removes the effect of perspective (brightness, shape) might be used in general (if possible).

For better Hough detection of the lane lines perhaps it would help to use some image filter to accentuate the typical colors of lanes, such as white or yellow.


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[solidWhiteCurve]: ./test_line_output/solidWhiteCurve.jpg
[solidWhiteRight]: ./test_line_output/solidWhiteRight.jpg

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

Easiest way for the computer to detect straigh lines is to first map all pixels from the image space to the Hough space. Ideally in the Hough space all pixels/points that are part of the same line will map to the same point, and those that are *close* to be on the same line will be near each other in the Hough space. So, the algorithm counts the number of points in the each *cell* of the Hough space, whose size will depend on the theta and rho resolution parameters.

**Interpolate detected lines to form lane lines**

When we have detected all straight lines in the image, we can *average* them in some way to form single line for each lane. In my solution I have written several functions which together are used in this process.

To average line *direction* I have first assumed that angle difference (in some preffered direction) between any two of those lines must lie in the range {0, pi/2}. If you position a set of lines to the origin (0,0) and then start calculating the angle difference from any chosen line to the others (in the same direction), then it follows this angle can be in the range (0, pi). In this general case it seems not possible to determine average direction just by averaging projections on any given coordinate system.

For average direction I have assumed there is one guiding line representing coordinate system to which all the lines (in polar coordinates) are mapped. Only the lines that are within (-pi/4, pi/4) of this line are considered for averaging, so the average direction can simply be calculated by averaging line normal vectors.

Average position is calculated by weighted sum of the line middle points, where the weights correspond to line lenghts. In this way we obtain the *center of mass* point of all the lines.

Some examples on how well average lines represent the lines derived from Hough:

![][solidWhiteCurve]
![][solidWhiteRight]

## Potential shortcomings with current pipeline

For detected line segments that are further away, perhaps it would make sense to assign weights to compensate the fact they must be visualy smaller. Similarly, the brightness of the road parts that are further away is expectedly lesser. These effects are not compensated.

I was a bit dissapointed in the results obtained from cv2.Canny(...) function, but I was not able to significantly improve the result by tweaking the parameters.

## Possible improvements

Perhaps some transformation that removes the effect of perspective (brightness, shape) might be used in general (if possible).

For calculating average direction for line segments perhaps it would be better to calculate line angles instead averaging projections, but that would require the use of expensive inverse functions ...


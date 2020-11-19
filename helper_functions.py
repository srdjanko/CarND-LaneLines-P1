import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

import globals

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns hough lines.
    """
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def hough_lines_image(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def polar_line_boundaries(polar_line, boundaries = None):
    """
    Input `polar_line` should contain: (rho, cos_theta, sin_theta)

    Returns line points in cartesian space from the polar space parameters for that line:
        (x1, y1, x2, y2)

    `boundaries` if provided define the image boundaries over which the segment should be presented.
    """

    x1, y1, x2, y2 = 0, 0, 0, 0
    rho, cos_theta, sin_theta, d = polar_line

    # Provide more numerical robustness by dividing over greater
    # projection for the line normal
    if cos_theta >= abs(sin_theta):
        y1 = 0

        if boundaries:
            y2 = boundaries[0]
        else:
            y2 = 1

        x1 = (rho - sin_theta * y1) / cos_theta
        x2 = (rho - sin_theta * y2) / cos_theta
    else:
        x1 = 0

        if boundaries:
            x2 = boundaries[1]
        else:
            x2 = 1

        y1 = (rho - cos_theta * x1) / sin_theta
        y2 = (rho - cos_theta * x2) / sin_theta

    return (x1, y1, x2, y2)

def cartesian_2_polar(line):
    """
    Input `line` should contain: (x1, y1, x2, y2)

    Transform the line to polar space using the typical definition:
        -pi/2 < theta < pi/2

    This allows for negative rho but also ensures: cos(theta) >= 0

    Returns:
        (rho, cos_theta, sin_theta, d)
    """

    x1, y1, x2, y2 = line
    dy = y2 - y1
    dx = x2 - x1

    return segment_2_polar(x1, y1, dx, dy)

def segment_2_polar(x1, y1, dx, dy):
    """
    Inputs represent line segment with one point and projections to (x, y) axis.
    Only the ratio of projections is actualy used in calculating polar line
    coordinates.

    Transform the line to polar space using the typical definition:
        -pi/2 < theta < pi/2

    This allows for negative rho but also ensures: cos(theta) >= 0

    Returns:
        (rho, cos_theta, sin_theta, d)
    """

    # Calculate the length of the line segment
    d = np.sqrt(dy*dy + dx*dx)

    cos_theta = -dy / d
    sin_theta = dx / d
    rho = y1*sin_theta + x1 * cos_theta

    # Ensure cos_theta >= 0 to comply with standard definition
    if cos_theta == 0 and sin_theta < 0:
        rho, sin_theta = -rho, -sin_theta

    elif cos_theta < 0:
        rho, cos_theta, sin_theta = -rho, -cos_theta, -sin_theta

    return (rho, cos_theta, sin_theta, d)

def calculate_average_sums(lines, middle_x):
    """
    Inputs
    - line segments used to detect left and right lane lines
    - middle x point that separates left and right lanes

    Outputs
    - sum of projection lengths
    - weighted sum of projection middle points. Weights represent lengths
      of projections
    """

     # { "x_sum": 0, "y_sum":0, "dx_sum":0, "dy_sum": 0 }
    left_lane = np.array([0.0, 0.0, 0.0, 0.0])
    right_lane = np.array([0.0, 0.0, 0.0, 0.0])

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # To keep it simple we disregard vertical lines
        if x2 == x1:
            continue

        elif x2 < x1:
            x1, x2 = x2, x1

        # Also disregard horizontal lines
        if y2 == y1:
            continue

        dx_sum = x2 - x1
        dy_sum = y2 - y1

        x_sum = (x1 + x2) * dx_sum / 2
        y_sum = (y2 + y1) * abs(dy_sum) / 2

        if (x1 < middle_x and x2 < middle_x) and (dy_sum < 0):
            left_lane += [x_sum, y_sum, dx_sum, dy_sum]

        elif (x1 > middle_x and x2 > middle_x) and (dy_sum > 0):
            right_lane += [x_sum, y_sum, dx_sum, dy_sum]

    return np.array([left_lane, right_lane])

def find_average_line(lines, middle_x, boundaries = None):
    """
    Inputs
    - lines obtained from Hough transform
    - middle x point that separates left and right lanes
    - image boundaries for plotting detected lane lines

    Outputs
    - detected left and right lanes

    """
    # { "x_sum": 0, "y_sum":0, "dx_sum":0, "dy_sum": 0 }
    current_lane = calculate_average_sums(lines, middle_x)

    # Average with previous values (Average with priors)
    current_lane = (current_lane + globals.glob_previous_lanes) / 2

    # Extend lines for by first converting to polar coordinates
    x_sum, y_sum, dx_sum, dy_sum = current_lane[0]

    # Remember, for left lane the dy projections are negative!
    mid_x, mid_y = x_sum / dx_sum, - y_sum / dy_sum
    polar_l = segment_2_polar(mid_x, mid_y, dx_sum, dy_sum)

    x_sum, y_sum, dx_sum, dy_sum = current_lane[1]
    mid_x, mid_y = x_sum / dx_sum, y_sum / dy_sum
    polar_r = segment_2_polar(mid_x, mid_y, dx_sum, dy_sum)

    # Remember sums for current lane, for later averaging
    globals.glob_previous_lanes = current_lane

    lane_lines = [(polar_line_boundaries(polar_l, boundaries), polar_line_boundaries(polar_r, boundaries))]

    return lane_lines

def lane_detection(image, params, lines_only = False):
    """
    Inputs
    - image to process lane lines
    - parameters for the lane detection algorithm
    - plot lines only or the combined image
    """

    # Gray scalling
    gray = grayscale(image)

    # Gaussian smoothing
    blur_gray = gaussian_blur(gray, params["kernel_size"])

    # Canny edge detection
    edges = canny(blur_gray, params["low_threshold"], params["high_threshold"])

    # Region of interest for lane detection as a four sided polygon
    imshape = image.shape

    left_boundary = params["left_boundary"]
    right_boundary = params["right_boundary"]
    lane_boundaries = np.array([left_boundary + right_boundary], dtype=np.int32)
    region_lines = region_of_interest(edges, lane_boundaries)

    # Hough transform
    h_lines = hough_lines(
        region_lines,
        params["rho_resolution"],
        params["theta_resolution"],
        params["min_votes"],
        params["min_line_length"],
        params["max_line_gap"])

    # Calculate middle point to separate lanes
    middle_x = (left_boundary[1][0] + right_boundary[0][0]) / 2
    avg_lines = find_average_line(h_lines, middle_x, imshape)

    # Create left and right lane lines
    left_lane = np.int32(avg_lines[0][0])
    right_lane = np.int32(avg_lines[0][1])

    lane_image = np.zeros_like(image)
    draw_lines(lane_image, np.array([[left_lane]]), color = [255,0,0], thickness = 5)
    draw_lines(lane_image, np.array([[right_lane]]), color = [255,0,0], thickness = 5)

    lane_image = region_of_interest(lane_image, lane_boundaries)

    if lines_only:
        draw_lines(lane_image, h_lines, color = [0,0,255])
        processed_image = lane_image
    else:
        processed_image = cv2.addWeighted(lane_image, 1, image, 0.6, 0)

    return processed_image

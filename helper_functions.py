import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

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

def polar_2_cartesian(polar_line, boundaries = None):
    """
    Input `polar_line` should contain: (rho, cos_theta, sin_theta)

    Returns line points in cartesian space from the polar space parameters for that line:
        (x1, y1, x2, y2)

    `boundaries` if provided define the image boundaries over which the line should be presented.
    """
    x1, y1, x2, y2 = 0, 0, 0, 0
    rho, cos_theta, sin_theta = polar_line

    # Provide more numerical robustness by dividing over greater projection for the line normal
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

def find_average_line(lines, guide_line, phi):
    """
    Finds the "average" line from the given line set that are also within the
    angle phi from the guide_line. The purpose of this is to filter out lines that
    deviate too much from the guide. Other reason is to confine the definition of
    "average" only for the lines that are within pi/2 degrees of each other, so it
    also follows that phi <= pi/4.

    Returns:
        (rho_avg, cos_theta_avg, sin_theta_avg)
    """

    # Calculate guide_line cos_theta, sin_theta
    (rho_g, cos_g, sin_g, d_g) = cartesian_2_polar(guide_line)

    # Max allowed deviation
    cos_phi = np.cos(phi)

    d_sum = 0
    median_x = 0
    median_y = 0
    cos_theta_g_sum = 0
    sin_theta_g_sum = 0

    for line in lines:
        (rho, cos_theta, sin_theta, d) = cartesian_2_polar(line[0])

        # Calculate coordinates referenced to guide_line by using scalar vector product
        # (cos_theta + I * sin_theta) * (cos_g - I * sin_g)
        cos_theta_g = cos_theta * cos_g + sin_theta * sin_g
        sin_theta_g = -cos_theta * sin_g + sin_theta * cos_g

        # Use abs since we are not interested in the sign of the projection
        if abs(cos_theta_g) >= cos_phi:
            x1, y1, x2, y2 = line[0]
            d_sum += d

            # Calculate the "center of mass" for the line segment to establish average position
            median_x += d * (x2 + x1)/2
            median_y += d * (y2 + y1)/2

            # Calculate sum of projections (for the line normal)
            cos_theta_g_sum += d * cos_theta_g
            sin_theta_g_sum += d * sin_theta_g

    # Finaly, calculate averages
    median_x = median_x / d_sum
    median_y = median_y / d_sum

    r = np.sqrt(sin_theta_g_sum * sin_theta_g_sum + cos_theta_g_sum * cos_theta_g_sum)
    cos_theta_avg = cos_theta_g_sum / r
    sin_theta_avg = sin_theta_g_sum / r

    # Map average line to original coordinate system (1, 0)
    # (cos_theta_avg + I * sin_theta_avg) * (cos_g + I * sin_g)
    cos_theta = cos_theta_avg * cos_g - sin_theta_avg * sin_g
    sin_theta = cos_theta_avg * sin_g + sin_theta_avg * cos_g

    # Finaly we get the average in polar coordinates, since we have pojection of the
    # line normal and one (middle) point
    rho_avg = cos_theta * median_x + sin_theta * median_y

    return (rho_avg, cos_theta, sin_theta)

def lane_segments(image, params, lines_only = False):
     # Gray scalling
    gray = grayscale(image)

    # Gaussian smoothing
    blur_gray = gaussian_blur(gray, params["kernel_size"])

    # Canny edge detection
    edges = canny(blur_gray, params["low_threshold"], params["high_threshold"])

    # Region of interest for lane detection as a four sided polygon
    left_boundary = params["left_boundary"]
    right_boundary = params["right_boundary"]
    lane_boundaries = np.array([left_boundary + right_boundary], dtype=np.int32)

    masked_edges = region_of_interest(edges, lane_boundaries)

    # Hough transform
    lines = hough_lines(
        masked_edges,
        params["rho_resolution"],
        params["theta_resolution"],
        params["min_votes"],
        params["min_line_length"],
        params["max_line_gap"])

    line_image = np.zeros_like(image)
    draw_lines(line_image, lines, color = [255,0,0])
    line_image = region_of_interest(line_image, lane_boundaries)

    return line_image if lines_only else cv2.addWeighted(line_image, 1, image, 0.6, 0)

def lane_detection(image, params, lines_only = False):

    # Gray scalling
    gray = grayscale(image)

    # Gaussian smoothing
    blur_gray = gaussian_blur(gray, params["kernel_size"])

    # Canny edge detection
    edges = canny(blur_gray, params["low_threshold"], params["high_threshold"])

    # Region of interest for lane detection as a four sided polygon
    imshape = image.shape
    middle_x = imshape[1]/2

    left_boundary = params["left_boundary"]
    right_boundary = params["right_boundary"]
    boundary_height = params["boundary_height"]

    left_lane_boundaries = np.array([left_boundary + [(middle_x, boundary_height), (middle_x, imshape[0])]], dtype=np.int32)
    right_lane_boundaries = np.array([[(middle_x, imshape[0]), (middle_x, boundary_height)] + right_boundary], dtype=np.int32)
    lane_boundaries = np.array([left_boundary + right_boundary], dtype=np.int32)

    masked_edges_left = region_of_interest(edges, left_lane_boundaries)
    masked_edges_right = region_of_interest(edges, right_lane_boundaries)

    # Hough transform
    hough_lines_left = hough_lines(
        masked_edges_left,
        params["rho_resolution"],
        params["theta_resolution"],
        params["min_votes"],
        params["min_line_length"],
        params["max_line_gap"])

    hough_lines_right = hough_lines(
        masked_edges_right,
        params["rho_resolution"],
        params["theta_resolution"],
        params["min_votes"],
        params["min_line_length"],
        params["max_line_gap"])

    # Guiding line, choosen as left and right region boundaries
    avg_line_left = find_average_line(hough_lines_left, left_boundary[0] + left_boundary[1], params["theta_deviation"])
    avg_line_right = find_average_line(hough_lines_right, right_boundary[0] + right_boundary[1], params["theta_deviation"])

    # Create left and right lane lines
    left_lane = np.int32(polar_2_cartesian(avg_line_left, imshape))
    right_lane = np.int32(polar_2_cartesian(avg_line_right, imshape))

    lane_image = np.zeros_like(image)
    draw_lines(lane_image, np.array([[left_lane]]), color = [255,0,0], thickness = 5)
    draw_lines(lane_image, np.array([[right_lane]]), color = [255,0,0], thickness = 5)

    lane_image = region_of_interest(lane_image, lane_boundaries)

    if lines_only:
        draw_lines(lane_image, hough_lines_left, color = [0,0,255])
        draw_lines(lane_image, hough_lines_right, color = [0,0,255])
        processed_image = lane_image
    else:
        processed_image = cv2.addWeighted(lane_image, 1, image, 0.6, 0)

    return processed_image


def plot_polar_lines(image, polar_lines):

    cartesian_lines = []

    for polar_line in polar_lines:
        rho, cos_theta, sin_theta = polar_line[0], np.cos(polar_line[1]), np.sin(polar_line[1])
        cartesian_lines.append(polar_2_cartesian([rho, cos_theta, sin_theta], image.shape))

    draw_lines(image, np.array([cartesian_lines], dtype=np.int32))

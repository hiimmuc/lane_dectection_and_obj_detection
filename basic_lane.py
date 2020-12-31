import numpy as np
import cv2
import matplotlib.pyplot as plt
DEFAULT_SIZE = (720, 480)


def display(input, r, c):
    '''
    plot images by mathplotlib
    '''
    plt.figure()
    for i, img in enumerate(input):
        plt.subplot(r, c, i+1), plt.imshow(img, 'gray'), plt.axis('off')
    plt.show()


def filter_colors(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # Filter white pixels
    lower_white = np.array([0, 130, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 0, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 1.)

    return image2


def preprocess_img(img):
    '''
    preprocess image before start process
    '''
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    mask = img

    # mask = cv2.cvtColor(mask, cv2.COLOR_HLS2BGR_FULL)
    cv2.imshow("filter", mask)

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (7, 7), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    mask = cv2.bitwise_and(img, img, mask=th)
    cv2.imshow("thresh", th)
    # smooth and sharpen image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    mask = cv2.bilateralFilter(mask, 7, 75, 75)
    mask = cv2.filter2D(mask, -1, kernel)
    cv2.imshow("bilarate", mask)
    mask = filter_colors(mask)  # enhance the value of white and yellow
    output = mask

    cv2.imshow("after", output)
    return [output]
    pass


def region_of_interest(image):
    '''
    mask the region that we focus on to detect line
    '''
    mask = np.zeros_like(image)
    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # We could have used fixed numbers as the vertices of the polygon,
    # but they will not be applicable to images with different dimesnions.
    height, width = image.shape[:2]

    bottom_left = [width * 0.1, height * 0.9]
    upper_left = [width * 0.04, height * 0.8]
    top_left = [width * 0.4, height * 0.6]
    bottom_right = [width * 0.9, height * 0.9]
    upper_right = [width * 0.96, height * 0.8]
    top_right = [width * 0.6, height * 0.6]

    vertices = np.array(
        [[bottom_left,   top_left, top_right,   bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    cv2.imshow("mask", masked_image)
    return masked_image


def drawLines(image, lines, line_color, line_thickness):
    ''' draw line :v '''

    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(image)
    # Checks if any lines are detected
    for x1, y1, x2, y2 in lines:
        # Draws lines between two coordinates with green color and 5 thickness
        cv2.line(lines_visualize, (x1, y1), (x2, y2),
                 line_color, line_thickness)
        # apply the mask to main image
    return cv2.addWeighted(image, 1, lines_visualize, 0.8, 1)
    pass


def average_slope_intercept(image, lines, old_line=[[], []]):
    '''find average slope of nearby line'''
    left_fit = []
    right_fit = []
    # h, w, _ = image.shape
    if (lines is None):
        return
    elif (len(lines) == 0):
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            # It will fit the polynomial and the intercept and slope
            parameters = np.polyfit((x1, x2), (y1, y2), deg=1)  # -> m and b
            slope = parameters[0] if x1 != x2 else 999
            intercept = parameters[1]
            # slope = (y2 - y1) / (x2 - x1)
            # intercept = y1 - (slope * x1)
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # if new value is much more diff from last value, we keep last value
    old_line_param = []
    if len(old_line[0]) != 0:
        for x1, y1, x2, y2 in old_line:
            parameters = np.polyfit((x1, x2), (y1, y2), deg=1)
            print("param", parameters)
            old_line_param.append(parameters)

    try:
        if len(old_line_param) != 0:
            left_fit_average = condition(left_fit_average, old_line_param[0])
            right_fit_average = condition(right_fit_average, old_line_param[1])
    except IndexError:
        pass
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = cal_coordinates(image, left_fit_average)
    right_line = cal_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])


def condition(new_line, old_line):
    print("old and new: ", old_line, new_line)
    if len(old_line) != 0 or old_line != None:
        cond1 = bool((new_line[0] - old_line[0]) >= 0.3 * old_line[0])
        cond2 = bool((new_line[1] - old_line[1]) >= 0.3 * old_line[1])
        if cond1 or cond2:
            return old_line
    return new_line


def cal_coordinates(image, params):
    '''convert slope and intercept to x,y coordinate'''
    try:
        slope, intercept = params
    except TypeError:
        slope, intercept = 0.001, 0

    # Sets initial y-coordinate as height from top down (bottom of the image)
    y1 = int(image.shape[0] * 0.97)
    # Sets final y-coordinate above the bottom of the image
    y2 = int(y1 * 0.8)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def get_car(car_path, image, pos):
    car = cv2.imread(car_path)
    car = cv2.resize(car, (100, 100))

    # I want to put logo on top-left corner, So I create a ROI
    h, w, _ = car.shape

    roi = image[pos[0] - h:pos[0], pos[1] - w//2:pos[1] + w//2, ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask = img2gray
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(car, car, mask=mask_inv)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)

    image[pos[0] - h//2:pos[0] + h//2, pos[1] - w//2:pos[1] + h//2] = dst
    return image, pos


def predict_movement(image, avg_lines, last_pos=(50, 50)):
    # predict movement by print out -1 0 1
    output = image
    h, w, _ = image.shape
    avg_lines = np.array(avg_lines)
    # compare init pos(last pos) and present pos to determine direction
    new_pos = (0, 0)
    dir_ = (avg_lines[0][0] + avg_lines[1][0]) // 2 - last_pos[1]
    cond = avg_lines[0][0] >= h or avg_lines[0][2] >= w * 0.4
    # draw obj in lane
    if last_pos == (50, 50) or cond:
        new_pos = last_pos
    else:
        new_pos = [(avg_lines[0][1] + avg_lines[1][1]) // 2 - 50,
                   (avg_lines[0][0] + avg_lines[1][0]) // 2]
    output, pos_prev = get_car('car.png', image=image, pos=new_pos)
    # print pred dir
    order = '0'
    if dir_ != 0:
        order = '1' if dir_ > 0 else '-1'
    print(order)
    #
    return output, pos_prev, order
    pass


def write_order(order, mode='a'):
    f = open(r'direction.txt', mode)
    f.writelines(f"{order}\n")
    f.close()
    pass


def test_on_video(path):
    ''' test on testing video '''
    # Path of dataset directory
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    outVid = cv2.VideoWriter(r'outvid.mp4', cv2.VideoWriter_fourcc(
        'D', 'I', 'V', 'X'), 15, (frame_width, frame_height))
    old_line = [[], []]
    last_pos = (0, 0)
    write_order("start\n", "w+")
    while(cap.isOpened()):
        _, frame = cap.read()
        after_preprocess = preprocess_img(frame)
        # display(after_preprocess, 2, 2)
        used_img = frame = after_preprocess[-1]

        gray = cv2.cvtColor(used_img, cv2.COLOR_BGR2GRAY)

        blur_gray = cv2.GaussianBlur(gray, (9, 9), 0)

        edges = cv2.Canny(blur_gray, 50, 150)
        # _, edges = cv2.threshold(blur_gray, 130, 145, cv2.THRESH_BINARY)
        cropped_image = region_of_interest(edges)

        lines = cv2.HoughLinesP(cropped_image,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=40,
                                lines=np.array([]),
                                minLineLength=20,
                                maxLineGap=1e2)

        averaged_lines = average_slope_intercept(frame, lines, old_line)
        old_line = averaged_lines
        # print(averaged_lines)
        try:
            output = drawLines(frame, averaged_lines, (0, 150, 0), 10)
        except OverflowError:
            output = frame
        # output, last_pos, order = predict_movement(
        # output, averaged_lines, last_pos=last_pos)

        outVid.write(output)
        cv2.imshow("result", output)
        # write_order(order)
        # # wait 0 will wait for infinitely between each frames.
        # 1ms will wait for the specified time only between each frames

        if cv2.waitKey(1) == 27:  # esc
            break

    # close the video file
    cap.release()
    outVid.release()
    # destroy all the windows that is currently on
    cv2.destroyAllWindows()
    pass


def test_on_img(path):
    ''' test on testing image '''
    image = cv2.imread(path)
    image = cv2.resize(image, DEFAULT_SIZE)
    after_preprocess = preprocess_img(image)
    # display(after_preprocess, 2, 2)

    used_img = after_preprocess[-1]
    cv2.imshow("after", used_img)
    gray = cv2.cvtColor(used_img, cv2.COLOR_BGR2GRAY)

    blur_gray = cv2.GaussianBlur(gray, (9, 9), 0)

    edges = cv2.Canny(blur_gray, 50, 150)
    cv2.imshow("edge", edges)
    cropped_image = region_of_interest(edges)
    cv2.imshow("cropped_img", cropped_image)
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 100,
                            np.array([]),
                            minLineLength=10,
                            maxLineGap=1e3)

    averaged_lines = average_slope_intercept(image, lines)
    output = drawLines(image, averaged_lines, (0, 150, 0), 10)  # green line
    cv2.imshow("results", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def run(path):
    if path[path.index('.')::] in ['.jpg', '.png', '.jpeg']:
        print('picture mode')
        test_on_img(path)
    else:
        print('video mode')
        test_on_video(path)


if __name__ == "__main__":
    '''make color :v'''
    path = r'F:\Lab Robotics&AI\day3\test_files\input.mp4'
    run(path)
    pass

# F:\Lab Robotics&AI\day3\test_files\input.mp4
# F:\Lab Robotics&AI\day1\test_videos\input.mp4
# need to fix OverflowError: Python int too large to convert to C long
# drawbacks: curve, high speed

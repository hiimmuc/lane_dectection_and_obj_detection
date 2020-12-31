
import time
import numpy as np
import cv2
from calibration import calib, undistort
import threshold
from finding_lines import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map

left_line = Line()
right_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (
    35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()


def process_frame(frame):
    img = frame
    # print(img.shape)
    undist_img = undistort(img, mtx, dist)
    undist_img = cv2.resize(
        undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    rows, cols = undist_img.shape[:2]
    # crop and compute gradient
    combined_gradient = threshold.gradient_combine(
        undist_img, th_sobelx, th_sobely, th_mag, th_dir)
    combined_hls = threshold.hls_combine(undist_img, th_h, th_l, th_s)
    combined_result = threshold.comb_result(combined_gradient, combined_hls)

    c_rows, c_cols = combined_result.shape[:2]
    s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
    s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(180, 720), (180, 0), (550, 0), (550, 720)])

    warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))

    searching_img = find_LR_lines(warp_img, left_line, right_line)
    w_comb_result, w_color_result = draw_lane(
        searching_img, left_line, right_line)

    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(
        w_color_result, Minv, (c_cols, c_rows))
    comb_result = np.zeros_like(undist_img)
    comb_result[220:rows - 12, 0:cols] = color_result

    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, comb_result, 0.3, 0)
    info, info2 = np.zeros_like(result),  np.zeros_like(result)
    info[5:110, 5:190] = (255, 255, 255)
    info2[5:110, cols-111:cols-6] = (255, 255, 255)

    info = cv2.addWeighted(result, 1, info, 0.5, 0)
    info2 = cv2.addWeighted(info, 1, info2, 0.5, 0)

    road_map = print_road_map(w_color_result, left_line, right_line)

    info2[10:105, cols-106:cols-11] = road_map
    info2, order = print_road_status(info2, left_line, right_line)
    # cv2.imshow('road info', info2)

    out_frame = info2
    return out_frame, order


def test_video(path):
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        _, frame = cap.read()
        t1 = time.time()
        output, _ = process_frame(frame=frame)
        cv2.imshow('road info', output)
        t2 = time.time()
        fps = int(1/(t2-t1))
        t1 = t2
        print('FPS: ', fps)
        # out.write(frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    pass


def run(path):
    if path[path.index('.')::] in ['.jpg', '.png', '.jpeg']:
        print('picture mode')
        output = process_frame(path=path)[0]
        cv2.imshow('result', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('video mode')
        test_video(path)


if __name__ == "__main__":
    '''make color :v'''
    path = r'F:\Lab Robotics&AI\day2\git_clone\advanced-lane-detection\videos\project_video.mp4'
    run(path)
    pass

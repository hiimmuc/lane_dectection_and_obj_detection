import time
import os
import threading
import obj_detect
import advanced_lane
import cv2
import numpy as np
import concurrent.futures
from multiprocessing.pool import ThreadPool
from threading import Thread


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


def combine(frame1, frame2, alpha, beta, gramma):
    return cv2.addWeighted(frame1, alpha, frame2, beta, gramma)


label = r"F:\Lab Robotics&AI\day3\backup\obj.names"
config = r"F:\Lab Robotics&AI\day3\backup\yolov4-tiny-custom.cfg"
net_path = r"F:\Lab Robotics&AI\day3\backup\yolov4-tiny-custom_best.weights"
net = obj_detect.create_net(config, net_path)
myYolov4 = obj_detect.Yolov4(net, config=config, label=label)

test_path = r'F:\Lab Robotics&AI\day2\git_clone\advanced-lane-detection\videos\project_video.mp4'
save_path = r"F:\Lab Robotics&AI\day3\report"


def test(path_in, path_out):

    cap = cv2.VideoCapture(path_in)
    width, height = list(map(int, [cap.get(3), cap.get(4)]))

    out = cv2.VideoWriter(os.path.join(path_out, 'output_combine.mp4'), cv2.VideoWriter_fourcc(
        'd', 'i', 'v', 'x'), 15, (width, height))

    while cap.isOpened():
        _, frame = cap.read()
        t1 = time.time()
        output_obj_detect, output_line_detect = np.zeros_like(
            frame), np.zeros_like(frame)
        # output_line_detect, order = advanced_lane.process_frame(frame)
        # _, output_obj_detect = myYolov4.detector(frame, 0.5, 0.4)
        func1 = ThreadWithReturnValue(
            target=advanced_lane.process_frame, args=(frame,))
        func2 = ThreadWithReturnValue(target=myYolov4.detector, args=(
            frame, 0.5, 0.4,))

        func1.start()
        func2.start()
        output_line_detect, _ = func1.join()
        _, output_obj_detect = func2.join()
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     func1 = executor.submit(advanced_lane.process_frame, frame)
        #     output_line_detect, order = func1.result()
        #     func2 = executor.submit(
        #         myYolov4.detector, frame, 0.5, 0.4)
        #     _, output_obj_detect = func2.result()
        # pool = ThreadPool(processes=2)
        # _, output_line_detect = pool.apply(
        #     func=advanced_lane.process_frame, args=(frame,))

        # output_obj_detect, _ = pool.apply(
        #     func=myYolov4.detector, args=(frame, 0.5, 0.4,))

        final_frame = cv2.addWeighted(
            output_line_detect, 1., output_obj_detect, 1., 0.)
        # fps
        t2 = time.time()
        fps = round(1/(t2-t1))
        t1 = t2
        print(f"FPS: {fps}")

        # print out
        out.write(final_frame)
        cv2.imshow("result", final_frame)
        cv2.imshow("line_detect", output_line_detect)
        cv2.imshow("obj_detect", output_obj_detect)
        # print(order)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


test(test_path, save_path)

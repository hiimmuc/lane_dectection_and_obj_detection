import time
import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

from numpy.core.numeric import zeros_like


class Yolov4:
    def __init__(self, net, config, label) -> None:
        self.net = net
        self.config = config
        self.label = label

    def detector(self, image, confidence_threshold, nms_threshold) -> Any:
        image = cv2.resize(image, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        mask = np.zeros_like(image)

        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1]
                         for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(
            image, 1/255., (320, 320), [0, 0, 0], swapRB=True, crop=False)

        self.net.setInput(blob)
        layer_outputs = self.net.forward(output_layers)

        class_names = []
        with open(self.label, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        colors = np.random.uniform(0, 255, size=(len(class_names), 3))

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    centerX, centerY, width, height = list(
                        map(int, detection[0:4] * [w, h, w, h]))

                    top_leftX, top_leftY = int(
                        centerX - width/2), int(centerY - height/2)
                    width, height = int(width), int(height)

                    boxes.append([top_leftX, top_leftY, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, nms_threshold)
        self.num_obj = len(indices)

        list_coor = []

        # crop_scale = 0.05
        # if len(indices) > 0:
        #     for i in indices.flatten():
        #         x, y, w, h = boxes[i]
        #         x = abs(int(x - crop_scale*w))
        #         y = abs(int(y - crop_scale*h))
        #         w = abs(int((1 + 2*crop_scale)*w))
        #         h = abs(int((1 + 2*crop_scale)*h))

        #         list_coor.append((x, y, w, h))

        # list_coor = sorted(list_coor, key=lambda x: x[0])

        # draw boxes
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            mask, f"Total Objects: {self.num_obj}", (w//3, 25), font, 0.8, [0, 255, 255], 2, lineType=cv2.LINE_AA)
        # cv2.putText(
        #     image, f"Total Objects: {self.num_obj}", (w//3, 25), font, 0.8, [0, 0, 0], 2, lineType=cv2.LINE_AA)
        for i in range((len(boxes))):
            if i in indices:
                x, y, w, h = boxes[i]
                tag = f"{class_names[class_ids[i]]}:{round(confidences[i],2)}"
                color = random.choice(colors)
                # cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=2)
                # cv2.putText(image, tag, (x, y-5), font, 0.6,
                #             color, 1, lineType=cv2.LINE_AA)

                cv2.rectangle(mask, (x, y), (x+w, y+h), color, thickness=2)
                cv2.putText(mask, tag, (x, y-5), font, 0.6,
                            color, 1, lineType=cv2.LINE_AA)
                list_coor.append([x, y, w, h])
        # cv2.imshow("mask", mask)
        # out_img = np.copy(image)
        return [[x, y, w, h] for x, y, w, h in list_coor], mask
        pass

    def __str__(self) -> str:
        return f"number of obj: {self.num_obj}"

    def num(self):
        return self.num_obj

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        return img


def create_net(config, net_path):
    net = cv2.dnn.readNetFromDarknet(config, net_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO]: Done reading net!")
    return net


label = r"F:\Lab Robotics&AI\day3\backup\obj.names"
config = r"F:\Lab Robotics&AI\day3\backup\yolov4-tiny-custom.cfg"
net_path = r"F:\Lab Robotics&AI\day3\backup\yolov4-tiny-custom_best.weights"

test_img_path = r"F:\Lab Robotics&AI\day3\test_files\27094_3063d356a3a54cc3859537fd23c5ba9d_1539205710.jpeg"
test_video_path = r"F:\Lab Robotics&AI\day3\test_files\input.mp4"


net = create_net(config, net_path)
myYolo = Yolov4(net=net, config=config, label=label)

# test image


# def test_image(path):
#     test_img = myYolo.load_image(path)
#     t = time.time()
#     coor, output_img = myYolo.detector(test_img, 0.5, 0.4)
#     print(coor)
#     print(time.time() - t, "s")

#     cv2.imshow("res", output_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def test_video(path):
    # test video
    out_dir = r"F:\Lab Robotics&AI\day3\report"
    cap = cv2.VideoCapture(path)
    width, height = list(map(int, [cap.get(3), cap.get(4)]))
    out = cv2.VideoWriter(os.path.join(out_dir, 'output.mp4'), cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), 15, (width, height))

    t1 = time.time()
    fps = 0
    while cap.isOpened():
        _, frame = cap.read()
        fps2 = cap.get(cv2.CAP_PROP_FPS)
        h, w, _ = frame.shape
        coor, output_img = myYolo.detector(frame, 0.5, 0.4)

        t2 = time.time()
        fps = int(1/(t2-t1))
        t1 = t2
        print("FPS:", fps, fps2)

        out.write(output_img)
        cv2.imshow("result", output_img)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_video(test_video_path)

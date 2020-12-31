        func2 = ThreadWithReturnValue(target=myYolov4.detector, args=(
            frame, 0.5, 0.4,))
        func2.start()
        _, output_line_detect = func2.join()

        func1 = ThreadWithReturnValue(
            target=advanced_lane.process_frame, args=(frame,))
        func1.start()
        output_obj_detect, _ = func1.join()
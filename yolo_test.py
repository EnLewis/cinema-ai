'''
'''
import os
import argparse
import time
import numpy as np
import cv2 as cv



def yolo_test_image():
    '''
    this is a docstring
    '''

    #TODO: Make arg parse block for getting files in a more modular way
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    im_path = args["image"]
    print(im_path)
    image = cv.imread(im_path)
    yolo_path = ".\\yolo_coco\\"
    blob_scale_factor = 1/255.0
    user_confidence = 0.5
    threshold = 0.3

    # Init paths
    labels_path = os.path.sep.join([yolo_path, "coco.names"])
    weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_path, "yolov3.cfg"])

    # Get labels for the objects the model was trained on
    LABELS = open(labels_path).read().strip().split("\n")

    # Initialize a list of colour for each class
    np.random.seed(22)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    print("loading YOLO from disk...")
    net = cv.dnn.readNetFromDarknet(config_path, weights_path)

    # Get only the output layer names that we need
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    image = cv.resize(image, None, fx=0.4, fy=0.4)
    (H, W) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, blob_scale_factor, (416, 416), swapRB=True, crop=False)

    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv.imshow(str(n), img_blob)

    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(layer_names)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    class_ids = []
    for out in layer_outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > user_confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv.dnn.NMSBoxes(boxes, confidences, user_confidence, threshold)

    font = cv.FONT_HERSHEY_PLAIN
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            cv.putText(image, text, (x, y-5), font, 0.5, (0,0,0), 1)

    cv.imshow('image', image)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def yolo_test_video():
    '''
    this is a docstring
    '''

    #TODO: Make arg parse block for getting files in a more modular way
    vid_path = ".\\test_videos\\vid_test.mp4"
    yolo_path = ".\\yolo_coco\\"
    vid_output = "D:\\Documents\\_Projects\cinema-ai\\test_videos\\vidoutput.avi"
    blob_scale_factor = 1/255.0
    user_confidence = 0.5
    threshold = 0.3
    total = 0

    # Init paths
    labels_path = os.path.sep.join([yolo_path, "coco.names"])
    weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_path, "yolov3.cfg"])

    # Get labels for the objects the model was trained on
    LABELS = open(labels_path).read().strip().split("\n")

    # Initialize a list of colour for each class
    np.random.seed(22)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    print("loading YOLO from disk...")
    net = cv.dnn.readNetFromDarknet(config_path, weights_path)

    # Get only the output layer names that we need
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv.VideoCapture(vid_path)
    writer = None
    (W, H) = (None, None)

    while 1:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        blob = cv.dnn.blobFromImage(frame, blob_scale_factor, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        start = time.time()
        layer_outputs = net.forward(layer_names)
        end = time.time()

        boxes = []
        confidences = []
        class_ids = []
        for out in layer_outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > user_confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")

                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, user_confidence, threshold)

        font = cv.FONT_HERSHEY_PLAIN
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]

                color = [int(c) for c in COLORS[class_ids[i]]]
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
                cv.putText(frame, text, (x, y-5), font, 0.5, (255,255,255), 1)
        
        if writer is None:
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter(vid_output, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
        
        writer.write(frame)
    
    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()

if __name__ == '__main__':
    #yolo_test_image()
    yolo_test_video()
    print("Destroying Scripts")
    cv.destroyAllWindows()

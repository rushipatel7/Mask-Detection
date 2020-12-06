import cv2
import imutils, sys
import numpy as np
import json
import time



# cfg = './ocr yolo/yolov3-tiny-obj.cfg'

# cfg = 'yolov3.cfg'
# whts = 'yolov3_last.weights'

# Define the codec and create VideoWriter object 
# fourcc = cv2.VideoWriter_fourcc(*'XVID') 
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
# out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

cfg = './yolov3-tiny (2 classes)/yolov3-tiny-obj.cfg'
whts = './yolov3-tiny (2 classes)/yolov3-tiny-obj_last (11).weights'

#cfg = './Yolov3-tiny-new/yolov3-tiny-obj.cfg'
#whts = './Yolov3-tiny-new/yolov3-tiny-obj_last (2).weights'


# whts = './ocr yolo/yolov3-tiny-obj_19000.weights'
net = cv2.dnn.readNetFromDarknet(cfg, whts)

#with open("./Yolov3-tiny-new/coco.names", "r") as f:
with open("./yolov3-tiny (2 classes)/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] 
colors = np.random.uniform(0, 255, size=(len(classes), 3))


#cap = cv2.VideoCapture("http://117.198.96.71:90/nphMotionJpeg?Resolution=640x480&Quality=100")
cap = cv2.VideoCapture("m2.mp4")
#cap = cv2.VideoCapture("m2.mp4")






font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')


while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            j = 1
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w + 10, y + h + 10), (255, 255, 0), 2)

            # cv2.rectangle(frame, (x, y), (x + w, y + h), fix_color, 2)
            # cv2.rectangle(frame, (x, y), (x + w, y), (255,255,0), -1)
            # cv2.imwrite('./output/sdad.jpg', frame)
            # img = cv2.imread("./output/sdad.jpg")
            # if label == 'plate':
            #   crop_img = img[y:y + h + 10, x:x + w + 20]
            #  cv2.imshow("cropped", crop_img)
            # cv2.imwrite("cropped.jpeg", crop_img)
            # small = cv2.resize(crop_img, (0, 0), fx=0.5, fy=0.5)
            # cv2.imwrite("cropped.jpeg", small)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y), font, 1, (255, 255, 255), 2)
            # if confidence > 0.9:
            #     cv2.imshow("cropped", crop_img)
            #     cv2.imwrite("cropped.jpeg", crop_img)
            #     small = cv2.resize(crop_img, (0, 0), fx=0.3, fy=0.3)
            #     cv2.imwrite("cropped.jpeg", small)

            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
            cv2.imshow("Image", frame)
            key = cv2.waitKey(1)

            if key == 27:
                out = cv2.VideoWriter('output_m2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, size, False)
                break
cap.release()

# After we release our webcam, we also release the output 
out.release()

cv2.destroyAllWindows()

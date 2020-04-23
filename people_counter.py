import cv2
import numpy as np
import dlib

# initialise lists and ints
frames_between_detection = 30
counter = 0
timer = 0
currentID = 0
redetected = False
trackable_objects = {}

class trackableobject:
    def get_bbox(self):
        return self.bbox
    def __init__ (self, objectID, bbox, tracker):
        self.objectID = objectID
        self.bbox = bbox
        self.counted = False    
        self.detected = True
        self.tracker = tracker


CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
TIME = 30
config_path = "cfg/yolov3.cfg"
weights_path = "weights/yolov3.weights"

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:
    grabbed, image = cap.read()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    if timer == TIME:
        # reset timer
        timer = 0
        # reset detections
        for objectID in trackable_objects:
            trackable_objects[objectID].detected = False
        # detection
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(ln)
        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the object detections
            for detection in output:
                # extract the class id (label) and confidence (as a probability) of the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # filter out weak predictions
                if class_id != 0:
                        continue
                if confidence > CONFIDENCE:
                    # filter out everything apart from people
                    redetected = False
                    # create bounding box
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    # find corners of bounding box
                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (height / 2))
                    endX = int(centerX + (width / 2))
                    endY = int(centerY + (height / 2))
                    bbox = (startX, startY, width, height)
                    # checking existing objects
                    for objectID in trackable_objects:
                        # getting old bounding box
                        bbox_ = trackable_objects[objectID].get_bbox()
                        cent_x_ = int(bbox_[0]) + (int(bbox_[2]))/2
                        cent_y_ = int(bbox_[1]) + (int(bbox_[3]))/2
                        # comparing centroid
                        if (cent_x_ - 75) < centerX < (cent_x_ + 75) and (cent_y_ - 75) < centerY < (cent_y_ + 75):
                            currentID = objectID
                            redetected = True
                            break
                    # adding new objects
                    if redetected == False:
                        to_dict_length = len(trackable_objects) + 2
                        for x in range(1, to_dict_length):
                            if x in trackable_objects:
                                continue
                            else:
                                currentID = x
                                t = dlib.correlation_tracker()
                                rect = dlib.rectangle(startX, startY, endX, endY)
                                t.start_track(rgb, rect)
                                trackable_objects[currentID] = trackableobject(currentID, bbox, t)
                                break
                    trackable_objects[currentID].detected = True
    # track and display
    for objectID in trackable_objects:
        if trackable_objects[objectID].detected == False:
            del trackable_objects[objectID]
            break
    # loop over each of the objects
    for objectID in trackable_objects:
        # update the tracker and find position of the object
        t = trackable_objects[objectID].tracker
        t.update(rgb)
        pos = t.get_position()
        # unpack the position object
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        # COUNTING
        # find new centroid
        cent_x = startX + (endX-startX)/2
        cent_y = startY + (endY-startY)/2
        # find old centroid
        bbox_ = trackable_objects[objectID].get_bbox()
        cent_x_ = int(bbox_[0]) + (int(bbox_[2]))/2
        cent_y_ = int(bbox_[1]) + (int(bbox_[3]))/2
        # check if already counted and time after last detection phase
        if trackable_objects[objectID].counted == False:
            # check if moving right
            if cent_x_ < (cent_x - 100):
                trackable_objects[objectID].counted = True
                counter = counter + 1
            # check if moving left
            if cent_x_ > (cent_x + 100):
                trackable_objects[objectID].counted = True
                counter = counter - 1
                if counter < 0:
                    counter = 0
        # update dictionary
        trackable_objects[objectID].bbox = (startX, startY, abs(startX-endX), abs(startY-endY))
        # make coords positive
        startX = max(0, startX)
        startY = max(0, startY)
        # display everything
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, 'ID: {}'.format(str(objectID)), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        blur_region = image[startY:endY, startX:endX]
        blur = cv2.GaussianBlur(blur_region, (51,51), 0)       
        image[startY:endY, startX:endX] = blur
    # show count in corner
    cv2.putText(image, 'Count = {}'.format(counter), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    # show image
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    
    timer = timer + 1

cap.release()
cv2.destroyAllWindows()
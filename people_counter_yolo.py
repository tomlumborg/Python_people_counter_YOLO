"""
ENGR 300 - People detection and tracking

Created Febuary 18 2020
Last updated May 10 2020

Author Thomas Lumborg
"""
#INITIALISATION
import cv2
import numpy as np
import dlib
width = []
height = []
averageheight_diff = 10
averagewidth_diff = 10
counter = 50
timer = 0
currentID = 0
redetected = False
trackable_objects = {}
CONFIDENCE = 0.8
TIME = 15
# portal (x, x, y, y)
portal = (230, 290, 0, 70)
config_path = "cfg/yolov3.cfg"
weights_path = "weights/yolov3.weights"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
class trackableobject:
    def get_bbox(self):
        return self.bbox
    def __init__ (self, objectID, bbox, tracker):
        self.objectID = objectID
        self.bbox = bbox
        self.counted = False    
        self.detected = True
        self.tracker = tracker
        self.start = 'none'
vs = cv2.VideoCapture(0)
while True:
	# get next frame from video source
    (grabbed, frame) = vs.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (h, w) = frame.shape[:2]
    # DETECTION (every x frames new detection phase)
    if timer == TIME:
        # reset timer
        timer = 0
        # reset detections
        for objectID in trackable_objects:
            trackable_objects[objectID].detected = False
        # convert the frame to a blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True)
        # pass the blob through the network and obtain the detections
        net.setInput(blob)
        layer_outputs = net.forward(ln)
        # loop layer outputs
        for output in layer_outputs:
            # loop detections
            for detection in output:
                # extract score, class and confidence (percentage)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # filter out low predictions and non-person classes
                if class_id != 0:
                        continue
                if confidence > CONFIDENCE:
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
                        # get old bounding box
                        bbox_ = trackable_objects[objectID].get_bbox()
                        cent_x_ = int(bbox_[0]) + (int(bbox_[2]))/2
                        cent_y_ = int(bbox_[1]) + (int(bbox_[3]))/2
                        # comparing centroid
                        if (cent_x_ - int(averagewidth_diff)) < centerX < (cent_x_ + int(averagewidth_diff)) and (cent_y_ - int(averageheight_diff)) < centerY < (cent_y_ + int(averageheight_diff)):
                            currentID = objectID
                            t = trackable_objects[objectID].tracker
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            t.start_track(rgb, rect)
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
        del_track_obj = {**trackable_objects}
        for objectID in del_track_obj:
            if del_track_obj[objectID].detected == False:
                del trackable_objects[objectID]
        # finding average width and height
        if len(trackable_objects) > 0:
            width = [0]*len(trackable_objects)
            height = [0]*len(trackable_objects)
            for i, objectID in enumerate(trackable_objects):
                boundingbox = trackable_objects[objectID].get_bbox()
                width[i] = int(boundingbox[2])
                height[i] = int(boundingbox[3])
            sum_ = 0
            for x in width:
                sum_ += x
            averagewidth_diff = (sum_ / (len(width)*4))
            sum_ = 0
            for x in height:
                sum_ += x
            averageheight_diff = (sum_ / (len(height)*4))
        
    else:    
    	# tracking
        for objectID in trackable_objects:
    		# update the tracker and grab the position of the object
            t = trackable_objects[objectID].tracker
            t.update(rgb)
            pos = t.get_position()
    		# unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # find centroid
            cent_x = startX + (endX-startX)/2
            cent_y = startY + (endY-startY)/2
            # update dictionary
            trackable_objects[objectID].bbox = (startX, startY, abs(startX-endX), abs(startY-endY))       
            # COUNTING
            # find starting location
            if trackable_objects[objectID].start == 'none':
                if int(portal[0]) < cent_x < int(portal[1]) and int(portal[2]) < cent_y < int(portal[3]):
                    trackable_objects[objectID].start = 'out'
                else:
                    trackable_objects[objectID].start = 'in'
            # if not counted check location
            elif trackable_objects[objectID].counted == False:
                if int(portal[0]) < cent_x < int(portal[1]) and int(portal[2]) < cent_y < int(portal[3]):
                    if trackable_objects[objectID].start == 'in':
                        counter -= 1
                        print('out', objectID)
                        trackable_objects[objectID].counted = True
                        if counter < 0:
                            counter = 0
                else:
                    if trackable_objects[objectID].start == 'out':
                        counter += 1
                        print('in', objectID)
                        trackable_objects[objectID].counted = True
    # display                    
    for objectID in trackable_objects:
        bbox = trackable_objects[objectID].get_bbox()
        startX = int(bbox[0])
        startY = int(bbox[1])
        endX = int(bbox[0]+bbox[2])
        endY = int(bbox[1]+bbox[3])                
        # make coords positive
        startX = max(0, startX)
        startY = max(0, startY)
        # display everything
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
        cv2.putText(frame, ('ID: {}'.format(str(objectID))), (startX, startY + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1)
        cv2.putText(frame, ('Start: {}'.format(str(trackable_objects[objectID].start))), (startX, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1)
        cv2.putText(frame, ('Counted: {}'.format(str(trackable_objects[objectID].counted))), (startX, startY + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1)
        blur_region = frame[startY:endY, startX:endX]
        blur = cv2.GaussianBlur(blur_region, (51,51), 0)       
        frame[startY:endY, startX:endX] = blur
    cv2.rectangle(frame, (0, 360), (640, int(h-30)), (0, 0, 0), -1)
    cv2.rectangle(frame, (int(portal[0]), int(portal[2])), (int(portal[1]), int(portal[3])), (0, 0, 0), 4)
    cv2.putText(frame, 'COUNT = {}'.format(counter), (int((w/2)-100), int(h-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    timer = timer + 1
vs.release()
cv2.destroyAllWindows()
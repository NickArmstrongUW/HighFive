import cv2
import time
import numpy as np
import rospy
from moveit_msgs.msg import MoveiTErrorCodes
from moveit_python import MoveGroupInterface, PlanningSceneInterface

def main():
    rospy.init_node("robo_highfive")

    ### Move it Init stuff ------------------------------------------------------------
    # Create move group interface for a fetch robot
    move_group = MoveGroupInterface("arm_with_torso", "base_link")

    # Define ground plane
    # This creates objects in the planning scene that mimic the ground
    # If these were not in place gripper could hit the ground
    planning_scene = PlanningSceneInterface("base_link")
    planning_scene.removeCollisionObject("my_front_ground")
    planning_scene.removeCollisionObject("my_back_ground")
    planning_scene.removeCollisionObject("my_right_ground")
    planning_scene.removeCollisionObject("my_left_ground")
    planning_scene.addCube("my_front_ground", 2, 1.1, 0.0, -1.0)
    planning_scene.addCube("my_back_ground", 2, -1.2, 0.0, -1.0)
    planning_scene.addCube("my_left_ground", 2, 0.0, 1.2, -1.0)
    planning_scene.addCube("my_right_ground", 2, 0.0, -1.2, -1.0)

    # TF joint names
    joint_names = ["torso_lift_joint", "shoulder_pan_joint",
                   "shoulder_lift_joint", "upperarm_roll_joint",
                   "elbow_flex_joint", "forearm_roll_joint",
                   "wrist_flex_joint", "wrist_roll_joint"]


    protoFile = "hand/pose_deploy.prototxt"
    weightsFile = "hand/pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

    threshold = 0.2

    ### Hand Detection init stuff  ---------------------------------------------------------------
    input_source = "asl.mp4"
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    aspect_ratio = frameWidth/frameHeight

    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)

    vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    k = 0

    highFive_Threshhold = [30, 30, 30, 30, 30]

    ### High Five Detection ------------------------------------------------------------------------
    while 1:
        k+=1
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()

        print("forward = {}".format(time.time() - t))

        # Empty list to store the detected keypoints
        points = []

        finger1 = None
        finger2 = None
        highFive = True
        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold :
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # if i == 8:
                #     finger1 = [int(point[0]), int(point[1])]
                # if i == 12:
                #     finger2 = [int(point[0]), int(point[1])]
                #     #distance = np.sqrt((finger1[0] - finger2[0]) ** 2 + (finger1[1] - finger2[1]) ** 2)
                    #print(distance)
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else :
                points.append(None)
                if (i != 21):
                    highFive = False

        print("highFive??", highFive)
        if highFive:

            pinky = np.sqrt((points[20][0] - points[18][0]) ** 2 + (points[20][1] - points[18][1]) ** 2)
            ring = np.sqrt((points[16][0] - points[14][0]) ** 2 + (points[16][1] - points[14][1]) ** 2)
            bird = np.sqrt((points[12][0] - points[10][0]) ** 2 + (points[12][1] - points[10][1]) ** 2)
            index = np.sqrt((points[8][0] - points[6][0]) ** 2 + (points[8][1] - points[6][1]) ** 2)
            thumb = np.sqrt((points[4][0] - points[2][0]) ** 2 + (points[4][1] - points[2][1]) ** 2)

            # print("pinky: ", pinky)
            # print("ring: ", ring)
            # print("bird: ", bird)
            # print("index: ", index)
            # print("thumb: ", thumb)

            finger_dists = [pinky, ring, bird, index, thumb]
            print("Finger Distances: ", finger_dists)

            for j in range(5):
                if finger_dists[j] < highFive_Threshhold[j]:
                    print("False Positive!", j)
                    break
            print()
        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        print("Time Taken for frame = {}".format(time.time() - t))

        # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(frame, "Hand Pose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Output-Skeleton', frame)
        # cv2.imwrite("video_output/{:03d}.jpg".format(k), frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

        print("total = {}".format(time.time() - t))

        vid_writer.write(frame)

    vid_writer.release()

if __name__ == '__main__':
    main()
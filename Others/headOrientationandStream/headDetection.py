import sys
import cv2
import numpy as np


def webcam_face_detect(video_mode, nogui = False, cascasdepath = "/Users/justinsabate/ThesisPython/virtualenv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"):

    face_cascade = cv2.CascadeClassifier(cascasdepath)
    eye_cascade = cv2.CascadeClassifier("/Users/justinsabate/ThesisPython/virtualenv/lib/python3.10/site-packages/cv2/data/haarcascade_eye.xml")

    video_capture = cv2.VideoCapture(video_mode)
    num_faces = 0
    '''initialisations for live head orientation detection'''
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    text = "Model"
    org = (100, 200)
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    color = (0, 0, 255)  # (B, G, R)
    thickness = 1
    lineType = cv2.LINE_AA
    bottomLeftOrigin = False



    while True:
        ret, image = video_capture.read()

        if not ret:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30,30)
            )

        # print("The number of faces found = ", len(faces))
        num_faces = len(faces)

        '''modifications to try to print head orientation real time'''
        size = image.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        '''end of modifications'''

        if not nogui:
            for (x,y,w,h) in faces:

                cv2.rectangle(image, (x,y), (x+h, y+h), (0, 255, 0), 2)
                '''Modifications'''
                image_points = np.array([
                    (x+w/2, y+h/2),  # Nose tip
                    (x+w/2, y+h),  # Chin
                    (x+w/3, y+h*2/3),  # Left eye left corner
                    (x+w*2/3, y+h*2/3),  # Right eye right corner
                    (x+w/3, y+h/3),  # Left Mouth corner
                    (x+w*2/3, y+h/3)  # Right mouth corner
                ], dtype="double")

                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                              dist_coeffs)
                # print("Rotation Vector:\n {0}".format(rotation_vector))
                text = str(rotation_vector)
                print(str(np.rad2deg(rotation_vector[1])))

                # cv2.putText(image, text, org, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
                # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                #                                                  translation_vector, camera_matrix, dist_coeffs)

                # for p in image_points:
                #     cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
                #
                # p1 = (int(image_points[0][0]), int(image_points[0][1]))
                # p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                #
                # cv2.line(image, p1, p2, (255, 0, 0), 2)
                '''end of modifications'''


            cv2.imshow("Faces found", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()
    return num_faces


if __name__ == "__main__":
    if len(sys.argv) < 2:
        video_mode= 0
    else:
        video_mode = sys.argv[1]
    webcam_face_detect(video_mode)
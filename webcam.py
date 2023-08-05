import cv2
from dt_apriltags import Detector
from pid import PID
from april_tags import *

vid = cv2.VideoCapture(0)
h_pid = PID(150, 0, 0, 100)
v_pid = PID(-150, 0, 0, 100)
while True:
    ret, frame = vid.read()
    # frame = cv2.imread("april_frame1.jpg")
    height = frame.shape[0]
    width = frame.shape[1]
    x_center = width / 2
    y_center = height / 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = get_tags(gray)
    if len(tags) > 0:
        positions = get_positions(tags)
        errors = error_relative_to_center(positions, width, height)
        outputs = output_from_tags(errors, h_pid, v_pid)
        horizontal_power = outputs[0][0]
        vertical_power = outputs[1][0]
        frame = cv2.arrowedLine(frame, (int(x_center), int(y_center)), (int(x_center), int(vertical_power) + int(y_center)), (0, 255, 0), 5, tipLength=0.5)
        frame = cv2.arrowedLine(frame, (int(x_center), int(y_center)), (int(horizontal_power) + int(x_center), int(y_center)), (0, 255, 0), 5, tipLength=0.5)
        frame = render_tags(tags, frame)
     
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()

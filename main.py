import cv2
import numpy as np

cap = cv2.VideoCapture(0)

saturation = 0
while 1:  # 727
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)

    # define range of blue color in HSV
    lower_orange = np.array([15, 80, 80])
    upper_orange = np.array([40, 255, 255])
    lower_white = np.array([0, 0, 200])

    upper_white = np.array([255, 5, 255])
    saturation += .1
    #print(saturation)

    # Threshold the HSV image to get only blue colors
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # add masks
    result = 255 * (orange_mask + white_mask)
    result_mask = result.clip(0, 255).astype("uint8")

    # Bitwise-AND mask and original image
    img = cv2.bitwise_and(frame, frame, mask=result_mask)

        
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]

    # get contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("test", result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

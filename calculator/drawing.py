import numpy as np
import cv2

drawing = False

def draw_circle(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img[y: y + 10, x: x + 10] = 255
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False    # cv2.circle(img,(x,y),100,(255,0,0),-1)

img = np.zeros((128, 1280, 1), np.uint8)
cv2.namedWindow('Write number')
cv2.setMouseCallback('Write number', draw_circle)

while True:
    cv2.imshow('Write number',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        # print(cv2.resize(img, (28, 28)))
        cv2.imwrite('test_img.jpg', img)
        break
cv2.destroyAllWindows()

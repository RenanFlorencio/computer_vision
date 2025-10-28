import cv2

phone_url = "rtsp://100.118.7.80:8080/h264_ulaw.sdp"
cap = cv2.VideoCapture(phone_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

import cv2

#NOT WORKING IN GOPRO? WHY?
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Windows: tente também CAP_MSMF
# Para Linux, experimente cv2.CAP_V4L ou cv2.CAP_GSTREAMER

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    cv2.imshow("Feed da Câmera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

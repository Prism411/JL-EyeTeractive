import cv2
from gaze_tracking import GazeTracking  # Certifique-se de que GazeTracking está instalado corretamente

gaze = GazeTracking()
webcam = cv2.VideoCapture("videos/video1.mp4")

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    # Obter coordenadas da pupila esquerda
    left_pupil = gaze.pupil_left_coords()

    # Se a coordenada da pupila esquerda estiver disponível, desenhar um círculo
    if left_pupil:
        gaze_x, gaze_y = left_pupil
        cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 255, 0), -1)  # Desenha um círculo verde no ponto da pupila esquerda
        print(gaze_x,gaze_y)
    # Exibir o quadro com o ponto de gaze desenhado
    cv2.imshow("Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()

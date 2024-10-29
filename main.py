import cv2
import time

# Carrega os classificadores pré-treinados para rosto e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Substitua 'video1.mp4' pelo caminho do seu vídeo
video_path = 'video1.mp4'
cap = cv2.VideoCapture(video_path)

# Reduz a resolução do vídeo para metade da original
scale_factor = 0.5

# Variáveis para calcular FPS
prev_frame_time = 0
frame_count = 0
detect_interval = 5  # Detectar apenas a cada 5 quadros

while cap.isOpened():
    # Lê o próximo frame do vídeo
    ret, frame = cap.read()

    # Verifica se o frame foi lido corretamente
    if not ret:
        break

    # Reduz a resolução do frame
    frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

    # Calcula o tempo atual e o FPS
    curr_frame_time = time.time()
    fps = 1 / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = curr_frame_time

    # Converte o frame para escala de cinza para melhorar a detecção
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realiza a detecção apenas a cada N quadros
    if frame_count % detect_interval == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

    # Itera sobre os rostos detectados
    for (x, y, w, h) in faces:
        # Define a região de interesse (ROI) para o rosto
        face_roi = gray[y:y + h, x:x + w]

        # Detecta os olhos dentro da ROI do rosto, mas somente se a face foi detectada no quadro atual
        if frame_count % detect_interval == 0:
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)

        # Verifica se encontrou os dois olhos
        if len(eyes) >= 2:
            # Ordena os olhos para garantir que o primeiro seja o esquerdo
            eyes = sorted(eyes, key=lambda ex: ex[0])
            eye_1 = eyes[0]
            eye_2 = eyes[1]

            # Calcula as coordenadas para a ROI que inclui os dois olhos
            ex, ey = min(eye_1[0], eye_2[0]), min(eye_1[1], eye_2[1])
            ew = max(eye_1[0] + eye_1[2], eye_2[0] + eye_2[2]) - ex
            eh = max(eye_1[1] + eye_1[3], eye_2[1] + eye_2[3]) - ey

            # Desenha retângulo na região dos dois olhos no frame original
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

    # Exibe o FPS no frame principal
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Exibe o frame principal com as detecções
    cv2.imshow('Face and Eyes Detection', frame)

    # Sai do loop ao pressionar "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Incrementa o contador de quadros
    frame_count += 1

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()

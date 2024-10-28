import cv2
import dlib
import time

# Carrega o detector de face e preditor de marcos faciais
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Carrega o vídeo
cap = cv2.VideoCapture("video.mp4")

# Fator de redimensionamento para reduzir a resolução do frame
scale_factor = 0.5  # Reduz para 50% do tamanho original
frame_skip = 2      # Processa a cada 2 frames

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Processa a cada 'frame_skip' frames
    if frame_count % frame_skip == 0:
        # Marca o início do tempo de processamento do frame
        frame_start_time = time.time()

        # Redimensiona o frame para acelerar o processamento
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detecta faces
        faces = detector(gray)

        for face in faces:
            # Ajusta as coordenadas para o frame original
            face = dlib.rectangle(
                int(face.left() / scale_factor),
                int(face.top() / scale_factor),
                int(face.right() / scale_factor),
                int(face.bottom() / scale_factor)
            )

            # Detecta os marcos faciais no frame original
            landmarks = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face)

            # Define a ROI ao redor apenas dos olhos (excluindo o nariz)
            left_eye_x = landmarks.part(36).x
            right_eye_x = landmarks.part(45).x
            top_y = min(landmarks.part(36).y, landmarks.part(45).y) - 20  # Acima dos olhos
            bottom_y = max(landmarks.part(40).y, landmarks.part(47).y) + 10  # Abaixo dos olhos

            # Garante que a ROI está dentro dos limites do frame
            roi = frame[max(0, top_y):min(frame.shape[0], bottom_y),
                        max(0, left_eye_x - 20):min(frame.shape[1], right_eye_x + 20)]

            # Converte a ROI para escala de cinza (preto e branco)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)  # Converte de volta para 3 canais para exibir

            # Calcula o FPS e exibe na ROI
            fps = 1.0 / (time.time() - frame_start_time)
            cv2.putText(roi_gray, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Detecta pontos dos olhos e desenha na ROI em escala de cinza
            for i in range(36, 42):  # Olho esquerdo
                x, y = landmarks.part(i).x - (max(0, left_eye_x - 20)), landmarks.part(i).y - (max(0, top_y))
                cv2.circle(roi_gray, (x, y), 2, (0, 255, 0), -1)
            for i in range(42, 48):  # Olho direito
                x, y = landmarks.part(i).x - (max(0, left_eye_x - 20)), landmarks.part(i).y - (max(0, top_y))
                cv2.circle(roi_gray, (x, y), 2, (0, 255, 0), -1)

            # Exibe apenas a ROI com os olhos em preto e branco
            cv2.imshow("ROI - Eyes Only (Grayscale)", roi_gray)

    # Conta os frames e define o controle de saída
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calcula o FPS médio ao final
end_time = time.time()
average_fps = frame_count / (end_time - start_time)
print(f"FPS médio: {average_fps:.2f}")

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()

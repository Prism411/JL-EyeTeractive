import cv2
import mediapipe as mp
import numpy as np

# Inicializa o módulo de Face Mesh do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Carrega o vídeo
cap = cv2.VideoCapture("video1.mp4")

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Fator de redimensionamento para reduzir a resolução do frame
scale_factor = 0.5

# Posição dos índices de landmarks para o olho esquerdo no MediaPipe
left_eye_indices = [33, 160, 158, 133, 153, 144]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona o frame para facilitar o processamento
    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Processa o frame com o MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        # Cria uma máscara binária do tamanho do frame original
        mask = np.zeros_like(frame)

        for face_landmarks in results.multi_face_landmarks:
            # Coleta as coordenadas dos pontos ao redor do olho esquerdo
            eye_coordinates = []
            for idx in left_eye_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                eye_coordinates.append((x, y))

            # Desenha o polígono na máscara com as coordenadas dos pontos ao redor do olho
            cv2.fillPoly(mask, [np.array(eye_coordinates)], (255, 255, 255))

            # Aplica a máscara ao frame original para extrair apenas a área dentro do polígono
            cropped_eye = cv2.bitwise_and(frame, mask)

            # Exibe apenas a área recortada do olho
            cv2.imshow("Cropped Eye Region", cropped_eye)

    # Exibe o frame original
    cv2.imshow("Original Video Frame", frame)

    # Aguarda até que uma tecla seja pressionada e fecha as janelas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()

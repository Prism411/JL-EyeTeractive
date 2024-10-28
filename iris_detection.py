import cv2
import mediapipe as mp
import numpy as np

# Inicializa o módulo de Face Mesh do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Índices de landmarks para o contorno do olho e íris (MediaPipe Face Mesh)
left_eye_indices = [33, 160, 158, 133, 153, 144]  # Contorno do olho esquerdo
left_iris_indices = [469, 470, 471, 472]  # Contorno da íris do olho esquerdo


# Índices de landmarks para o contorno do olho e íris do olho direito no MediaPipe
right_eye_indices = [263, 362, 387, 373, 380, 385]  # Contorno do olho direito
right_iris_indices = [474, 475, 476, 477]  # Contorno da íris do olho esquerdo

# Carrega o vídeo
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

def get_cropped_eye(frame, eye_indices, iris_indices):
    """Recorta e retorna a região do olho e da íris em um frame."""
    # Cria uma máscara binária do tamanho do frame original
    mask = np.zeros_like(frame)

    # Redimensiona o frame para facilitar o processamento
    scale_factor = 0.5
    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Processa o frame com o MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Coleta as coordenadas dos pontos ao redor do olho
            eye_coordinates = []
            for idx in eye_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                eye_coordinates.append((x, y))

            # Coleta as coordenadas dos pontos ao redor da íris
            iris_coordinates = []
            for idx in iris_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                iris_coordinates.append((x, y))

            # Desenha o polígono para o contorno do olho na máscara
            cv2.fillPoly(mask, [np.array(eye_coordinates)], (255, 255, 255))

            # Desenha pequenos círculos na máscara para os pontos da íris
            for (x, y) in iris_coordinates:
                cv2.circle(mask, (x, y), 2, (255, 255, 255), -1)

            # Aplica a máscara ao frame original para extrair apenas a área dentro do polígono
            cropped_eye = cv2.bitwise_and(frame, mask)

            return cropped_eye  # Retorna a área recortada do olho

    return None  # Retorna None se não encontrar landmarks

# Carrega o vídeo
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtém a área recortada do olho direito e da íris
    cropped_eye = get_cropped_eye(frame, left_eye_indices, left_iris_indices)
    #cropped_eye = get_cropped_eye(frame, right_eye_indices, right_iris_indices)
    if cropped_eye is not None:
        # Exibe apenas a área recortada do olho direito e íris
        cv2.imshow("Cropped Right Eye and Iris Region", cropped_eye)

    # Exibe o frame original
    cv2.imshow("Original Video Frame", frame)

    # Aguarda até que uma tecla seja pressionada e fecha as janelas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()
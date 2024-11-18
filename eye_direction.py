import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from matrixHandler import calcular_centroide, atualizar_grafico, configurar_grafico, calcular_distancias, \
    calcular_graus, calcular_direcao_paraconsistente
# Inicializa o módulo de Face Mesh do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Índices de landmarks para o contorno do olho e íris (MediaPipe Face Mesh)
left_eye_indices = [33, 160, 158, 133, 153, 144]  # Contorno do olho esquerdo
left_iris_indices = [469, 470, 471, 472]  # Contorno da íris do olho esquerdo


# Índices de landmarks para o contorno do olho e íris do olho direito no MediaPipe
right_eye_indices = [263, 362, 387, 373, 380, 385 ]  # Contorno do olho direito
right_iris_indices = [474, 475, 476, 477]  # Contorno da íris do olho esquerdo

# Configura o gráfico interativo
fig, ax, eye_plot, iris_plot, centroid_plot = configurar_grafico()

# Carrega o vídeo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

def get_cropped_eye(frame, eye_indices, iris_indices):
    """Recorta e retorna a região do olho e da íris em um frame."""
    # Redimensiona o frame para processamento mais rápido
    scale_factor = 1
    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mask = np.zeros_like(frame)

    # Processa o frame redimensionado com MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            eye_coordinates = []
            for idx in eye_indices:
                x = int(face_landmarks.landmark[idx].x * small_frame.shape[1] / scale_factor)
                y = int(face_landmarks.landmark[idx].y * small_frame.shape[0] / scale_factor)
                eye_coordinates.append((x, y))

            iris_coordinates = []
            for idx in iris_indices:
                x = int(face_landmarks.landmark[idx].x * small_frame.shape[1] / scale_factor)
                y = int(face_landmarks.landmark[idx].y * small_frame.shape[0] / scale_factor)
                iris_coordinates.append((x, y))

            centroid = calcular_centroide(iris_coordinates)
            #centroid = gaze.pupil_left_coords()
            distancias = calcular_distancias(eye_coordinates, centroid)
            grau_horizontal, grau_vertical = calcular_graus(distancias)
            print(grau_vertical)
            print(calcular_direcao_paraconsistente(grau_horizontal, grau_vertical))
            cv2.fillPoly(mask, [np.array(eye_coordinates)], (255, 255, 255))

            for (x, y) in iris_coordinates:
                cv2.circle(mask, (x, y), 2, (255, 255, 255), -1)
                cv2.circle(mask, calcular_centroide(iris_coordinates), 2, (0, 0, 255), 5)

            cropped_eye = cv2.bitwise_and(frame, mask)
            #print(eye_coordinates)
            #print(iris_coordinates)

            return cropped_eye, eye_coordinates, iris_coordinates, centroid

    return None

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtém a área recortada do olho direito e da íris
    cropped_eye, eye_coords, iris_coords, iris_centroid = get_cropped_eye(frame, left_eye_indices, left_iris_indices)
    # Obtém a área recortada do olho direito e da íris
    #cropped_eye, eye_coords, iris_coords, iris_centroid = get_cropped_eye(frame, right_eye_indices, right_iris_indices)

    #cropped_eye = get_cropped_eye(frame, right_eye_indices, right_iris_indices)
    if cropped_eye is not None:
        # Exibe apenas a área recortada do olho direito e íris
        cv2.imshow("Cropped Right Eye and Iris Region", cropped_eye)
        # Atualiza o gráfico com os dados atuais
        atualizar_grafico(eye_plot, iris_plot, centroid_plot, eye_coords, iris_coords, iris_centroid)
    # Exibe o frame original
    cv2.imshow("Original Video Frame", frame)

    # Aguarda até que uma tecla seja pressionada e  fecha as janelas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
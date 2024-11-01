import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe FaceMesh e parâmetros de desenho
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Função para detectar o olho esquerdo, recortar o formato exato e aplicar limiarização e contornos
def process_eye(image):
    # Converter imagem para RGB, necessário para o MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    # Verifica se algum rosto foi detectado
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Coordenadas dos pontos do olho esquerdo
            left_eye_points = [
                face_landmarks.landmark[33],
                face_landmarks.landmark[133],
                face_landmarks.landmark[160],
                face_landmarks.landmark[159],
                face_landmarks.landmark[158],
                face_landmarks.landmark[144],
                face_landmarks.landmark[153],
                face_landmarks.landmark[145],
                face_landmarks.landmark[7]
            ]

            # Obter as coordenadas exatas do olho esquerdo
            h, w, _ = image.shape
            eye_coords = np.array([[int(point.x * w), int(point.y * h)] for point in left_eye_points])

            # Criar uma máscara do mesmo tamanho do frame
            mask = np.zeros((h, w), dtype=np.uint8)

            # Preencher o formato do olho na máscara
            cv2.fillPoly(mask, [eye_coords], 255)

            # Aplicar a máscara ao frame original para obter apenas o olho
            eye_only = cv2.bitwise_and(image, image, mask=mask)

            # Calcular a área delimitadora do olho com uma margem intermediária
            x_min, x_max = np.min(eye_coords[:, 0]) - 10, np.max(eye_coords[:, 0]) + 10
            y_min, y_max = np.min(eye_coords[:, 1]) - 10, np.max(eye_coords[:, 1]) + 10
            eye_crop = eye_only[y_min:y_max, x_min:x_max]

            # Converter para escala de cinza
            gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

            # Aplicar desfoque Gaussiano
            blurred_eye = cv2.GaussianBlur(gray_eye, (25, 25), 5)

            # Aplicar a limiarização binária inversa
            _, binary_eye = cv2.threshold(blurred_eye, 35, 255, cv2.THRESH_BINARY)

            # Detectar contornos
            contours, _ = cv2.findContours(binary_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Desenhar os contornos no recorte do olho em uma cor visível
            contour_image = cv2.cvtColor(binary_eye, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

            # Exibir o recorte do olho com contornos
            cv2.imshow("Recorte do Olho Esquerdo com Contornos", contour_image)

    return image

# Captura de vídeo
cap = cv2.VideoCapture("video1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Processar o frame para detectar e exibir o olho com contornos
    processed_frame = process_eye(frame)

    # Exibir o frame original processado (sem o contorno do olho no frame principal)
    cv2.imshow("Vídeo Original", processed_frame)

    # Pressionar 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()

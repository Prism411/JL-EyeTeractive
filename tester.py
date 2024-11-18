import cv2
import mediapipe as mp

# Inicializar MediaPipe FaceMesh e parâmetros de desenho
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Função para detectar o olho esquerdo, recortar, aplicar limiarização e detectar contornos
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

            # Definir área de recorte ao redor do olho
            h, w, _ = image.shape
            x_coords = [int(point.x * w) for point in left_eye_points]
            y_coords = [int(point.y * h) for point in left_eye_points]

            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

            # Recortar e converter para escala de cinza
            eye_crop = image[y_min:y_max, x_min:x_max]
            gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

            # Aplicar desfoque Gaussiano
            blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)

            # Aplicar a limiarização binária inversa
            _, binary_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)

            # Detectar contornos
            contours, _ = cv2.findContours(binary_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Desenhar os contornos na imagem original em uma cor visível
            contour_image = cv2.cvtColor(binary_eye, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

            # Exibir a imagem com contornos desenhados
            cv2.namedWindow("Contornos do Olho Esquerdo", cv2.WINDOW_NORMAL)
            cv2.imshow("Contornos do Olho Esquerdo", contour_image)

            # Aguardar até que uma tecla seja pressionada para fechar a janela
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Carregar a imagem
image = cv2.imread("photos/img_1.png")
process_eye(image)

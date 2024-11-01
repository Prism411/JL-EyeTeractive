import cv2
import dlib
import mediapipe as mp
import time

from matrixHandler import calcular_centroide, configurar_grafico, atualizar_grafico

# Inicializa o detector de face do dlib e o preditor de marcos faciais
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializa o módulo de soluções faciais do mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Carrega o vídeo
cap = cv2.VideoCapture("video2.mp4")

# Fator de redimensionamento para reduzir a resolução do frame
scale_factor = 0.5
frame_skip = 2
frame_count = 0
start_time = time.time()
screenshot_count = 0  # Para salvar múltiplas capturas com nomes únicos
fig, ax, eye_plot, iris_plot, centroid_plot = configurar_grafico()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Processa a cada 'frame_skip' frames
    if frame_count % frame_skip == 0:
        frame_start_time = time.time()

        # Redimensiona o frame
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detecta faces com o dlib
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

            # Usa o frame completo para a detecção com o mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Define a ROI ao redor do olho esquerdo usando o dlib
            left_eye_x_start = landmarks.part(36).x
            left_eye_x_end = landmarks.part(39).x
            left_top_y = min(landmarks.part(36).y, landmarks.part(39).y) - 20
            left_bottom_y = max(landmarks.part(37).y, landmarks.part(40).y) + 10

            # Coordenadas dos pontos da íris e do olho esquerdo
            iris_left_coordinates = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for i in range(469, 473):  # Índices da íris esquerda no Mediapipe
                        x = int(face_landmarks.landmark[i].x * frame.shape[1]) - (left_eye_x_start - 20)
                        y = int(face_landmarks.landmark[i].y * frame.shape[0]) - left_top_y
                        iris_left_coordinates.append((x, y))
                        cv2.circle(frame, (x + left_eye_x_start - 20, y + left_top_y), 3, (255, 0, 0), -1)
                print(f"Coordenadas Íris (Olho Esquerdo): {iris_left_coordinates}")
            centroide = calcular_centroide(iris_left_coordinates)
            # Calcula e destaca o centróide da íris
            if iris_left_coordinates:
                print("Centroid", centroide)
                cv2.circle(frame, (centroide[0] + left_eye_x_start - 20, centroide[1] + left_top_y), 1, (0, 255, 255),
                           -1)
            eye_left_coordinates = []
            for i in range(36, 42):
                x = landmarks.part(i).x - (left_eye_x_start - 20)
                y = landmarks.part(i).y - left_top_y
                eye_left_coordinates.append((x, y))
                cv2.circle(frame, (x + left_eye_x_start - 20, y + left_top_y), 2, (0, 255, 0), -1)
            print(f"Coordenadas Olho Esquerdo: {eye_left_coordinates}")

            # Exibe o frame completo com as marcações do olho esquerdo
            cv2.imshow("Eye and Iris Detection - Left Eye Only", frame)
            atualizar_grafico(eye_plot, iris_plot, centroid_plot, eye_left_coordinates, iris_left_coordinates,centroide)

    # Conta os frames e define o controle de saída
    frame_count += 1
    key = cv2.waitKey(1) & 0xFF

    # Verifica se a tecla 'j' foi pressionada
    if key == ord('j'):
        screenshot_filename = f"print_{screenshot_count}.png"
        cv2.imwrite(screenshot_filename, frame)
        print(f"Captura de tela salva como {screenshot_filename}")
        screenshot_count += 1  # Incrementa o contador para salvar múltiplas imagens

    # Encerra o loop ao pressionar 'q'
    if key == ord('q'):
        break

# Calcula o FPS médio ao final
end_time = time.time()
average_fps = frame_count / (end_time - start_time)
print(f"FPS médio: {average_fps:.2f}")

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()

import cv2
import dlib
import mediapipe as mp
import time

# Inicializa o detector de face do dlib e o preditor de marcos faciais
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializa o módulo de soluções faciais do mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Carrega o vídeo
cap = cv2.VideoCapture("video1.mp4")

# Fator de redimensionamento para reduzir a resolução do frame
scale_factor = 0.5
frame_skip = 2
frame_count = 0
start_time = time.time()
screenshot_count = 0  # Para salvar múltiplas capturas com nomes únicos

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

            # Define a ROI ao redor dos olhos usando o dlib
            left_eye_x = landmarks.part(36).x
            right_eye_x = landmarks.part(45).x
            top_y = min(landmarks.part(36).y, landmarks.part(45).y) - 20
            bottom_y = max(landmarks.part(40).y, landmarks.part(47).y) + 10

            # Garante que a ROI está dentro dos limites do frame
            roi = frame[max(0, top_y):min(frame.shape[0], bottom_y),
                  max(0, left_eye_x - 20):min(frame.shape[1], right_eye_x + 20)]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

            # Desenha a posição da íris se pontos forem detectados no rosto completo
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_landmarks = [face_landmarks.landmark[i] for i in range(474, 478)]
                    right_eye_landmarks = [face_landmarks.landmark[i] for i in range(469, 473)]

                    # Ajusta os pontos para a ROI e desenha na imagem
                    for landmark in left_eye_landmarks + right_eye_landmarks:
                        x = int(landmark.x * frame.shape[1]) - max(0, left_eye_x - 20)
                        y = int(landmark.y * frame.shape[0]) - max(0, top_y)
                        cv2.circle(roi_gray, (x, y), 2, (255, 0, 0), -1)

            # Desenha os pontos dos olhos com o dlib
            for i in range(36, 42):  # Olho esquerdo
                x, y = landmarks.part(i).x - (max(0, left_eye_x - 20)), landmarks.part(i).y - (max(0, top_y))
                cv2.circle(roi_gray, (x, y), 2, (0, 255, 0), -1)
            for i in range(42, 48):  # Olho direito
                x, y = landmarks.part(i).x - (max(0, left_eye_x - 20)), landmarks.part(i).y - (max(0, top_y))
                cv2.circle(roi_gray, (x, y), 2, (0, 255, 0), -1)

            # Calcula o FPS e exibe na ROI
            fps = 1.0 / (time.time() - frame_start_time)
            #cv2.putText(roi_gray, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Exibe apenas a ROI com os olhos em preto e branco
            cv2.imshow("ROI - Eyes with Iris Detection", roi_gray)

    # Conta os frames e define o controle de saída
    frame_count += 1
    key = cv2.waitKey(1) & 0xFF

    # Verifica se a tecla 'j' foi pressionada
    if key == ord('j'):
        screenshot_filename = f"print_{screenshot_count}.png"
        cv2.imwrite(screenshot_filename, roi_gray)
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

import cv2
import mediapipe as mp
import pyautogui
import utils
import random
from pynput.mouse import Button, Controller

# Inisialisasi kontrol mouse dan ukuran layar
mouse = Controller()
screen_width, screen_height = pyautogui.size()

# Inisialisasi MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Faktor smoothing dan variabel untuk posisi sebelumnya
smoothing_factor = 0.2
previous_x, previous_y = 0, 0

def find_finger_tip(processed):
    """ Menemukan ujung jari telunjuk dari hasil proses MediaPipe """
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(index_finger_tip):
    """ Menggerakkan mouse ke posisi yang dihaluskan """
    global previous_x, previous_y
    if index_finger_tip is not None:
        # Hitung posisi kursor dari landmark jari
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)

        # Smoothing: Menghaluskan gerakan kursor
        smoothed_x = int(previous_x + (x - previous_x) * smoothing_factor)
        smoothed_y = int(previous_y + (y - previous_y) * smoothing_factor)

        # Update posisi sebelumnya
        previous_x, previous_y = smoothed_x, smoothed_y

        # Pindahkan mouse ke posisi yang telah dihaluskan
        pyautogui.moveTo(smoothed_x, smoothed_y)

def is_left_click(landmarks_list, thumb_index_dist):
    """ Mengecek apakah gesture adalah klik kiri """
    return (
        utils.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        utils.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) > 90 and
        thumb_index_dist > 50
    )

def is_right_click(landmarks_list, thumb_index_dist):
    """ Mengecek apakah gesture adalah klik kanan """
    return (
        utils.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        utils.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90 and
        thumb_index_dist > 50
    )

def is_double_click(landmarks_list, thumb_index_dist):
    """ Mengecek apakah gesture adalah klik ganda """
    return (
        utils.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        utils.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        thumb_index_dist > 50
    )

def is_screenshot(landmarks_list, thumb_index_dist):
    """ Mengecek apakah gesture adalah screenshot """
    return (
        utils.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        utils.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        thumb_index_dist < 50
    )

def is_scroll_up(landmarks_list, thumb_index_dist):
    """ Mengecek apakah gesture adalah scroll ke atas """
    return (
        utils.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        thumb_index_dist < 50
    )

def is_scroll_down(landmarks_list, thumb_index_dist):
    """ Mengecek apakah gesture adalah scroll ke bawah """
    return (
        utils.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        thumb_index_dist < 50
    )

def detect_gestures(frame, landmarks_list, processed):
    """ Mendeteksi dan mengeksekusi aksi berdasarkan gesture tangan """
    if len(landmarks_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = utils.get_distance([landmarks_list[4], landmarks_list[5]])

        # Gerakkan mouse jika jarak jari telunjuk dan sudut sesuai
        if thumb_index_dist < 50 and utils.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90:
            move_mouse(index_finger_tip)

        # Klik kiri
        elif is_left_click(landmarks_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "left click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Klik kanan
        elif is_right_click(landmarks_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "right click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Klik ganda
        elif is_double_click(landmarks_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Screenshot
        elif is_screenshot(landmarks_list, thumb_index_dist):
            iml = pyautogui.screenshot()
            label = random.randint(1, 1000)
            iml.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Scroll up
        elif is_scroll_up(landmarks_list, thumb_index_dist):
              mouse.scroll(0, 1)  # Scroll up
              cv2.putText(frame, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Scroll Down
        elif is_scroll_down(landmarks_list, thumb_index_dist):
            mouse.scroll(0, -1)  # Scroll down
            cv2.putText(frame, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def main():
    """ Fungsi utama untuk menjalankan video capture dan pemrosesan gesture """
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    # Atur resolusi video capture untuk mengurangi beban pemrosesan
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Sesuaikan FPS sesuai dengan kebutuhan

    # Atur ukuran jendela yang akan muncul
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1024, 768)  # Mengatur ukuran window menjadi 1024x768 piksel

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmarks_list = []

            if processed.multi_hand_landmarks:
                hands_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hands_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hands_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))

            detect_gestures(frame, landmarks_list, processed)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

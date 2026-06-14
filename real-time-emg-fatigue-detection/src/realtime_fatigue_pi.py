import serial
import numpy as np
import json
import tensorflow as tf
from collections import deque
from collections import Counter

# =========================
# AYARLAR
# =========================
SERIAL_PORT = "/dev/ttyUSB0"   # gerekirse değiştir
BAUD_RATE = 115200

WINDOW_SIZE = 200
STEP_SIZE = 100

# Tahmin stabil olsun diye son N tahmini oylayacağız
SMOOTHING_SIZE = 5

# =========================
# MODELİ YÜKLE
# =========================
model = tf.keras.models.load_model("final_fatigue_cnn.keras")

with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}

print("Model ve label map yüklendi.")
print("Label map:", label_map)

# =========================
# BUFFER
# =========================
buffer = deque(maxlen=WINDOW_SIZE)
recent_preds = deque(maxlen=SMOOTHING_SIZE)

# =========================
# SERİ PORT
# =========================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print(f"Seri port açıldı: {SERIAL_PORT} @ {BAUD_RATE}")

# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def majority_vote(pred_list):
    if len(pred_list) == 0:
        return None
    return Counter(pred_list).most_common(1)[0][0]

def parse_serial_line(line):
    """
    Arduino'dan gelen satırı float'a çevirir.
    Arduino tarafı her satırda tek sayı göndermeli.
    Örn: 512.3
    """
    try:
        return float(line.strip())
    except:
        return None

# =========================
# ANA DÖNGÜ
# =========================
print("Gerçek zamanlı tahmin başladı...")

while True:
    try:
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        value = parse_serial_line(raw)

        if value is None:
            continue

        buffer.append(value)

        # 200 sample dolunca tahmin yap
        if len(buffer) == WINDOW_SIZE:
            window = np.array(buffer, dtype=np.float32)

            # Model eğitimde ham pencereyle çalıştı, burada ek normalizasyon yapmıyoruz
            x_input = window.reshape(1, WINDOW_SIZE, 1)

            pred_prob = model.predict(x_input, verbose=0)
            pred_idx = int(np.argmax(pred_prob, axis=1)[0])
            pred_label = label_map[pred_idx]
            confidence = float(np.max(pred_prob))

            recent_preds.append(pred_label)
            stable_label = majority_vote(recent_preds)

            print(f"Ham Tahmin: {pred_label:12s} | Güven: {confidence:.3f} | Stabil: {stable_label}")

            # 100 sample overlap için son 100 sample kalsın
            last_samples = list(buffer)[-STEP_SIZE:]
            buffer.clear()
            buffer.extend(last_samples)

    except KeyboardInterrupt:
        print("Durduruldu.")
        break

    except Exception as e:
        print("Hata:", e)
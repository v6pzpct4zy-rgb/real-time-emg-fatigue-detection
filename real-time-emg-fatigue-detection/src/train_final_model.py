import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================
# VERİYİ YÜKLE
# =========================
X = np.load("X_windows_trimmed.npy")
y = np.load("y_labels_trimmed.npy")
groups = np.load("groups_subject_trimmed.npy")  

# recovery varsa çıkar
mask = y != "recovery"
X = X[mask]
y = y[mask]
groups = groups[mask]

print("X shape:", X.shape)
print("y shape:", y.shape)

# CNN için reshape
X = X.reshape(X.shape[0], X.shape[1], 1)

# Label encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = list(le.classes_)

print("Class names:", class_names)

# Class weights
classes = np.unique(y_encoded)
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_encoded
)
class_weights = {int(i): float(w) for i, w in zip(classes, class_weights_array)}

print("Class weights:", class_weights)

# =========================
# MODEL
# =========================
model = Sequential([
    Input(shape=(X.shape[1], 1)),

    Conv1D(32, 5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    Conv1D(64, 5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.30),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-5,
    verbose=1
)

# =========================
# EĞİTİM
# =========================
history = model.fit(
    X, y_encoded,
    validation_split=0.1,
    epochs=25,
    batch_size=32,
    verbose=1,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# =========================
# KAYDET
# =========================
model.save("final_fatigue_cnn.keras")
print("Model kaydedildi: final_fatigue_cnn.keras")

# Label map kaydet
label_map = {int(i): cls for i, cls in enumerate(class_names)}
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("Label map kaydedildi: label_map.json")
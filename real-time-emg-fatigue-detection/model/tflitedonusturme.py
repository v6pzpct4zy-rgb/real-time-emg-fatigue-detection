import tensorflow as tf

model = tf.keras.models.load_model("final_fatigue_cnn.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model hazır!")
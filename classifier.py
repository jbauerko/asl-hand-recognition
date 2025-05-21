import tensorflow as tf
import numpy as np
import cv2

class TFLiteClassifier:
    def __init__(self, model_path, labels_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def predict(self, image):
        input_shape = self.input_details[0]['shape'][1:3]
        image_resized = cv2.resize(image, tuple(input_shape))
        image_normalized = image_resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(image_normalized, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        prediction = np.argmax(output)
        confidence = output[prediction]
        return self.labels[prediction], confidence

cap = cv2.VideoCapture(0)
classifier = TFLiteClassifier("./model/model.tflite", "./model/labels.txt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, confidence = classifier.predict(frame)
    cv2.putText(frame, f'{label} ({confidence:.2f})', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

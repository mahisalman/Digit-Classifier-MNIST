import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# ---------------------
# 1. Load and Preprocess Dataset
# ---------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# ---------------------
# 2. Build the Model
# ---------------------
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),

    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------
# 3. Train the Model
# ---------------------
history = model.fit(
    x_train, y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.2,
    verbose=2
)

# ---------------------
# 4. Plot Accuracy & Loss
# ---------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss over Epochs")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------
# 5. Evaluate on Test Set
# ---------------------
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save("mnist_digit_classifier.h5")

# ---------------------
# 6. GUI for Handwriting Input
# ---------------------
class DigitRecognizerApp:
    def __init__(self, model):
        self.model = model
        self.canvas_size = 380
        self.image_size = 28

        self.root = tk.Tk()
        self.root.title("Draw a Digit")

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.pack(side='left')

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side='left')

        self.label = tk.Label(self.root, text="Draw a digit and click Predict")
        self.label.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.label.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        img = self.image.resize((self.image_size, self.image_size))
        img = ImageOps.invert(img)
        img = np.array(img).astype('float32') / 255.0
        img = img.reshape(1, 28, 28)

        prediction = self.model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        self.label.config(text=f"Prediction: {digit} (Confidence: {confidence:.2f})")

    def run(self):
        self.root.mainloop()


# Load and launch the GUI
if __name__ == '__main__':
    loaded_model = load_model("mnist_digit_classifier.h5")
    app = DigitRecognizerApp(loaded_model)
    app.run()



model.save("mnist_digit_classifier.h5")

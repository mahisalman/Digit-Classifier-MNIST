# ✨ Digit Classifier – MNIST (Flask + TensorFlow)

A real-time **handwritten digit recognizer** powered by a trained **TensorFlow neural network** on the **MNIST dataset**.  
This project includes an interactive **HTML5 Canvas UI** and a **Flask API backend** to serve live predictions using a `.keras` model.

![Screenshot](https://github.com/mahisalman/Digit-Classifier-MNIST/blob/main/Digit-Classifier-MNIST.png)

---

## 🚀 Live Demo

🖼️ Coming soon or deploy locally (see below).

---

## 📦 Project Structure

digitclassifier/
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   ├── index.html
│   └── train.html
├── model/
│   └── mnist_model.h5
├── notebooks/
│   └── training.ipynb
├── app.py
├── model_builder.py
├── README.md
└── requirements.txt


---

## 🛠️ Features

- 🎨 Draw a digit (0–9) on canvas
- ⚙️ Preprocess and resize to 28×28
- 🧠 Predict using TensorFlow model
- 📊 Displays prediction + confidence
- 🌐 Flask API backend (`/predict`)

---

## 💡 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mahisalman/Digit-Classifier-MNIST.git
cd Digit-Classifier-MNIST


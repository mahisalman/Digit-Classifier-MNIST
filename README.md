# ✨ Digit Classifier – MNIST (Flask + TensorFlow)

A real-time **handwritten digit recognizer** powered by a trained **TensorFlow neural network** on the **MNIST dataset**.  
This project includes an interactive **HTML5 Canvas UI** and a **Flask API backend** to serve live predictions using a `.keras` model.

![Screenshot](https://github.com/mahisalman/Digit-Classifier-MNIST/blob/main/Digit-Classifier-MNIST.png)

---

## 🚀 Live Demo

🖼️ Coming soon or deploy locally (see below).

---

📁 Project Folder Structure
```
Digit-Classifier-MNIST/
├── templates/
│   └── index.html             # HTML5 canvas frontend (in templates folder)
│
├── Digit-Classifier-MNIST.png # Screenshot or demo image
├── README.md                  # Project documentation
├── app.py                     # Flask backend (web server)
├── mnist.py                   # Model training script
├── best_model.keras           # Best performing model (from training)
├── mnist_digit_classifier.h5  # Older HDF5 model (legacy format)
├── mnist_digit_classifier.keras # Final model for GUI & web use
├── requirements.txt           # Python dependencies
```


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


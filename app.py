from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==============================
# 1. CARREGANDO O MODELO
# ==============================
MODEL_PATH = "cats_vs_dogs_mobilenetv2.keras"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (160, 160)

# Pasta temporária para uploads
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==============================
# 2. FUNÇÃO DE PREDIÇÃO
# ==============================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        return f"🐶 Cão detectado com {prediction*100:.2f}% de confiança"
    else:
        return f"🐱 Gato detectado com {(1 - prediction)*100:.2f}% de confiança"


# ==============================
# 3. ROTAS
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    uploaded_image = None

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            prediction_text = predict_image(filepath)
            uploaded_image = filepath

    return render_template("index.html", prediction=prediction_text, image=uploaded_image)


# ==============================
# 4. RODAR O SERVIDOR
# ==============================
if __name__ == "__main__":
    app.run(debug=True)

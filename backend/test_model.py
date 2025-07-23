# test_model.py

from model import load_model, preprocess_image, predict

model = load_model()
img = preprocess_image("data/test_image.jpg")
label, conf = predict(model, img)

print(f"Prediction: {label} ({conf*100:.2f}%)")

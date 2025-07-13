import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os

folder_path ='dataset/Sample/Suara Ayam Filipin Berkokok.wav'
output_folder = 'test'
filename = "Sample.png"
# Buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

def save_spectogram_image(wav_path, image_path):
    y, sr = librosa.load(wav_path, sr=22050)
    plt.figure(figsize=(3, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(os.path.join(image_path,filename), bbox_inches='tight', pad_inches=0)
    plt.close()

save_spectogram_image(folder_path, output_folder)

# Load model
model = tf.keras.models.load_model('cnn_model_ayam.h5')

# Load LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load gambar spectogram untuk prediksi
img = image.load_img(os.path.join(output_folder,filename), target_size=(96,96))
img_array = image.img_to_array(img) / 255
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
pred= model.predict(img_array)
pred_index = np.argmax(pred)
pred_label = le.inverse_transform([pred_index])[0]

print("Classify Predict : ",pred_label)
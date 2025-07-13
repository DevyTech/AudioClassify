import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os

folder_path ='dataset/Sample/Suara Ayam Cemani Berkokok.wav'
output_folder = 'test'
filename = "Sample.png"
# Buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

def save_mfcc_image(wav_path, image_path):
    y, sr = librosa.load(wav_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(image_path,filename), bbox_inches='tight', pad_inches=0)
    plt.close()

save_mfcc_image(folder_path, output_folder)

# Load model
model = tf.keras.models.load_model('model/model_final.h5')

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
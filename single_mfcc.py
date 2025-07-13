import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

folder_path ='dataset/Sample'
output_folder = 'mfcc_images/test'
# Buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

def save_mfcc_image(wav_path, image_path):
    y, sr = librosa.load(wav_path, sr=22050)

    plt.figure(figsize=(3,3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

save_mfcc_image('dataset/Sample/Suara Ayam Bangkok Berkokok.wav','mfcc_images/test/tesbangkok.png')
print("Job Done")
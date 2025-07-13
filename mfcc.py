import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Path ke folder
dataset_path = 'dataset/bangkok'
output_folder_mfcc = 'mfcc_images/bangkok'
output_folder_spectogram = 'spectogram_images/bangkok'

# Buat folder output kalau belum ada
os.makedirs(output_folder_mfcc, exist_ok=True)
os.makedirs(output_folder_spectogram, exist_ok=True)

# List untuk menyimpan path gambar spectogram & mfcc
spectograms_path = []
mfcc_paths = []

def extract_spectogram_mfcc_image(audio_path, save_name):
    y, sr = librosa.load(audio_path, sr=22050)

    # SPEKTOGRAM
    plt.figure(figsize=(3,3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectogram : {save_name}")
    plt.tight_layout()
    spect_path = os.path.join(output_folder_spectogram, f"{save_name}_spectogram.png")
    plt.savefig(spect_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Tambahkan ke list path spectograms
    spectograms_path.append(spect_path)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    plt.figure(figsize=(3,3))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.axis('off')
    plt.tight_layout()
    mfcc_path = os.path.join(output_folder_mfcc, f"{save_name}_mfcc.png")
    plt.savefig(mfcc_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Tambahkan ke list path mfcc
    mfcc_paths.append(mfcc_path)

# Loop semua file .wav dalam folder
for filename in os.listdir(dataset_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(dataset_path, filename)
        output_name = filename.replace('.wav', '')
        print(f"Memproses: {filename}")
        extract_spectogram_mfcc_image(file_path, output_name)

print("Semua spektrogram & MFCC berhasil disimpan")

print("Tampilkan spectogram")
img_spec = Image.open(spectograms_path[10])
plt.figure(figsize=(4,4))
plt.imshow(img_spec)
plt.axis('off')
plt.tight_layout()
plt.show()

print("Tampilkan mfcc")
img_mfcc = Image.open(mfcc_paths[10])
plt.figure(figsize=(4,4))
plt.title("MFCC")
plt.imshow(img_mfcc)
plt.axis('off')
plt.tight_layout()
plt.show()

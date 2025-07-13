import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle


# Membuat objek ImageDataGenerator dengan split untuk validasi dan normalisasi piksel gambar
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

# Mengambil data pelatihan dari folder dan membagi sesuai subset 'training'
train_data = datagen.flow_from_directory(
    'mfcc_images/',
    target_size=(96,96), # Ukuran gambar diubah ke 96x96 piksel
    class_mode='categorical', # Label dikonversi ke format one-hot
    subset='training', # Ambil data training
    shuffle=True
)

# Mengambil data validasi dari folder dan membagi sesuai subset 'validation'
val_data = datagen.flow_from_directory(
    'mfcc_images/',
    target_size=(96,96),
    class_mode='categorical',
    subset='validation', # Ambil data validasi
    shuffle=False
)

# Mendapatkan daftar nama kelas dari struktur folder
class_labels = list(train_data.class_indices.keys())
print("Label yang digunakan : ", class_labels)

# Membuat dan melatih LabelEncoder dengan nama-nama kelas
le = LabelEncoder()
le.fit(class_labels)

# Menyimpan LabelEncoder ke file .pkl
with open('label_encoder.pkl','wb') as f:
    pickle.dump(le, f)

# Membuat model CNN sederhana
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)), # Layer konvolusi pertama
    tf.keras.layers.MaxPooling2D(2, 2), # Layer max pooling

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Layer konvolusi kedua
    tf.keras.layers.MaxPooling2D(2, 2), # Layer max pooling kedua

    tf.keras.layers.Flatten(), # Mengubah tensor 2D menjadi 1D
    tf.keras.layers.Dense(64, activation='relu'), # Fully connected layer
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax') # Output layer dengan jumlah neuron sesuai jumlah kelas
])

# Menyusun model dengan optimizer Adam dan loss function untuk klasifikasi multi-kelas
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model menggunakan data training dan validasi
model.fit(train_data, validation_data=val_data, epochs=10)

# Menyimpan model yang telah dilatih ke file
model.save('model/cnn_model_ayam.h5')
print('Model dan LabelEncoder berhasil disimpan.')

# Amber prediksi dari data validasi
val_data.reset() # reset generator
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)

# Ambil label asli dari generator
y_true = val_data.classes

# Tampilkan confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_data.class_indices.keys())

disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Validasi")
plt.show()
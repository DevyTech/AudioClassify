import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from glob import glob
from PIL import Image

# ==== KONFIGURASI ====
data_dir = 'mfcc_images/'
img_size = (96, 96)
batch_size = 16
num_folds = 5
epochs = 10

# Ambil semua file dan label
class_names = sorted(os.listdir(data_dir))
all_image_paths = []
all_labels = []

for class_name in class_names:
    folder_path = os.path.join(data_dir, class_name)
    for img_file in glob(f"{folder_path}/*.png"):
        all_image_paths.append(img_file)
        all_labels.append(class_name)

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(all_labels)

# Simpan label encoder untuk dipakai saat prediksi
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Ubah gambar ke array
def load_images(img_paths):
    images = []
    for path in img_paths:
        img = Image.open(path).resize(img_size).convert('RGB')
        images.append(np.array(img) / 255.0)
    return np.array(images)

X = load_images(all_image_paths)
y = np.array(y_encoded)

# ==== K-FOLD TRAINING ====
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold = 1
all_accuracies = []

for train_idx, val_idx in kf.split(X):
    print(f"\nFold {fold}/{num_folds}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # One-hot encoding label
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(class_names))

    # Buat model CNN
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Simpan model terbaik dari tiap fold
    checkpoint_path = f"model/model_fold{fold}.h5"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
    )

    model.fit(X_train, y_train_cat, epochs=epochs, validation_data=(X_val, y_val_cat),
              callbacks=[checkpoint_cb], verbose=1)

    # Evaluasi
    model.load_weights(checkpoint_path)
    val_pred = np.argmax(model.predict(X_val), axis=1)
    acc = np.mean(val_pred == y_val)
    all_accuracies.append(acc)

    print(f"Akurasi Fold {fold}: {acc:.4f}")
    print("\n", classification_report(y_val, val_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_val, val_pred)
    plt.figure()
    plt.title(f"Confusion Matrix Fold {fold}")
    plt.imshow(cm, cmap='Blues')
    plt.xlabel("Prediksi")
    plt.ylabel("Asli")
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"conf_matrix/conf_matrix_fold{fold}.png")
    plt.close()

    fold += 1

print("\n============================")
print("Rata-rata Akurasi:", np.mean(all_accuracies))
print("============================")

# ==== FINAL TRAINING DENGAN SELURUH DATA ====
print("\nTraining akhir dengan seluruh data...")
y_cat = tf.keras.utils.to_categorical(y, num_classes=len(class_names))

final_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
final_model.fit(X, y_cat, epochs=epochs, verbose=1)
final_model.save("model/model_final.h5")
print("Model akhir disimpan sebagai model_final.h5")
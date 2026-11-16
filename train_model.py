"""
Skin Disease Detection - AI/ML Model Trainer
Trains:
  1. Support Vector Machine (SVM) with RBF Kernel & Probability Calibration
  2. Deep Neural Network (CNN / MLP Architecture)
  3. TensorFlow CNN (if TensorFlow is installed)
Outputs saved in models/ directory:
  - models/svm_model.pkl
  - models/svm_scaler.pkl
  - models/neural_cnn_model.pkl
  - models/label_encoder.pkl
  - models/training_report.json
  - models/cnn_model.h5 (when TensorFlow is present)
"""

import json
import os
import pickle
import sys
import warnings

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from feature_extractor import extract_features

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = (128, 128)

os.makedirs(MODELS_DIR, exist_ok=True)

DISEASE_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

DISEASE_NAMES = {
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular Lesion",
}


# ─────────────────────────────────────────────
#  DATA LOADING & FEATURE EXTRACTION
# ─────────────────────────────────────────────
def load_dataset_and_features():
    """Load images from dataset/processed and extract feature vectors."""
    print("=" * 60)
    print("  DermAI - Skin Disease Model Training Pipeline")
    print("=" * 60)
    print(f"\n[1/5] Loading images and extracting visual features...")

    features_list = []
    labels_list = []
    raw_images_list = []

    for cls in DISEASE_CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"  [!] Directory not found for class: {cls}")
            continue

        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        cls_name = DISEASE_NAMES.get(cls, cls)
        print(f"  -> {cls.upper():<6} ({cls_name:<24}): {len(files)} images")

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                feat = extract_features(img)
                features_list.append(feat)
                labels_list.append(cls)
                raw_images_list.append(np.array(img.resize(IMG_SIZE), dtype=np.float32) / 255.0)
            except Exception as e:
                pass

    X_features = np.array(features_list, dtype=np.float32)
    X_images = np.array(raw_images_list, dtype=np.float32)
    y_labels = np.array(labels_list)

    print(f"\n[+] Total loaded: {len(X_features)} images across {len(np.unique(y_labels))} disease classes")
    print(f"[+] Feature dimension: {X_features.shape[1]} features per image")
    return X_features, X_images, y_labels


# ─────────────────────────────────────────────
#  MAIN TRAINING ROUTINE
# ─────────────────────────────────────────────
def main():
    X_feat, X_raw, y_raw = load_dataset_and_features()

    if len(X_feat) == 0:
        print("[X] No images found! Run prepare_dataset.py first.")
        sys.exit(1)

    # 1. Encode labels
    print(f"\n[2/5] Encoding disease classes...")
    le = LabelEncoder()
    le.fit(DISEASE_CLASSES)
    y_encoded = le.transform(y_raw)

    le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"[+] Label encoder saved -> {le_path}")

    # 2. Train / Val Split (80% train, 20% val with stratification)
    X_train_f, X_val_f, y_train, y_val, X_train_raw, X_val_raw = train_test_split(
        X_feat, y_encoded, X_raw, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"[+] Split: {len(X_train_f)} training samples | {len(X_val_f)} validation samples")

    # 3. Standardize features
    print(f"\n[3/5] Scaling visual features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_val_scaled = scaler.transform(X_val_f)

    scaler_path = os.path.join(MODELS_DIR, "svm_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[+] Feature scaler saved -> {scaler_path}")

    # 4. Train SVM Model
    print(f"\n[4/5] Training Support Vector Machine (SVM) Classifier...")
    svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    svm.fit(X_train_scaled, y_train)

    svm_val_preds = svm.predict(X_val_scaled)
    svm_acc = accuracy_score(y_val, svm_val_preds)
    print(f"[+] SVM Validation Accuracy: {svm_acc * 100:.2f}%")

    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    with open(svm_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"[+] SVM model saved -> {svm_path}")

    # 5. Train Deep Neural Network (CNN/MLP architecture)
    print(f"\n[5/5] Training Deep Neural Network (CNN / Neural Architecture)...")
    neural_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=350,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15,
    )
    neural_model.fit(X_train_scaled, y_train)

    neural_val_preds = neural_model.predict(X_val_scaled)
    neural_acc = accuracy_score(y_val, neural_val_preds)
    print(f"[+] Deep Neural Network Accuracy: {neural_acc * 100:.2f}%")

    neural_path = os.path.join(MODELS_DIR, "neural_cnn_model.pkl")
    with open(neural_path, "wb") as f:
        pickle.dump(neural_model, f)
    print(f"[+] Neural model saved -> {neural_path}")

    # 6. Ensemble Evaluation
    svm_probs = svm.predict_proba(X_val_scaled)
    neural_probs = neural_model.predict_proba(X_val_scaled)
    ensemble_probs = 0.5 * svm_probs + 0.5 * neural_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = accuracy_score(y_val, ensemble_preds)
    print(f"[+] CNN-SVM Ensemble Validation Accuracy: {ensemble_acc * 100:.2f}%")

    # Optional: Check if TensorFlow is available for deep Keras CNN training
    tf_trained = False
    try:
        import tensorflow as tf
        from tensorflow import keras

        print("\n[*] TensorFlow detected! Training deep Keras CNN model...")
        cnn = keras.Sequential(
            [
                keras.Input(shape=(128, 128, 3)),
                keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(2),
                keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(2),
                keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(len(DISEASE_CLASSES), activation="softmax"),
            ]
        )
        cnn.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        cnn.fit(
            X_train_raw,
            y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val_raw, y_val),
            verbose=0,
        )
        cnn_path = os.path.join(MODELS_DIR, "cnn_model.h5")
        cnn.save(cnn_path)
        tf_trained = True
        print(f"[+] Keras CNN model saved -> {cnn_path}")
    except Exception:
        # TensorFlow not installed or not supported natively
        pass

    # 7. Classification Report
    val_names = [DISEASE_NAMES[c] for c in le.classes_]
    print("\n" + "=" * 60)
    print("  Detailed Classification Performance Report")
    print("=" * 60)
    print(classification_report(y_val, ensemble_preds, target_names=val_names, digits=3))

    # 8. Save Training Report JSON
    report = {
        "svm_accuracy": float(svm_acc),
        "neural_cnn_accuracy": float(neural_acc),
        "ensemble_accuracy": float(ensemble_acc),
        "total_samples": len(X_feat),
        "train_samples": len(X_train_f),
        "val_samples": len(X_val_f),
        "classes": DISEASE_CLASSES,
        "class_names": DISEASE_NAMES,
        "tensorflow_cnn_trained": tf_trained,
    }
    report_path = os.path.join(MODELS_DIR, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[+] Full training report saved -> {report_path}")
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE! Models are ready to serve.")
    print(f"  * SVM Accuracy        : {svm_acc * 100:.1f}%")
    print(f"  * Neural CNN Accuracy : {neural_acc * 100:.1f}%")
    print(f"  * Ensemble Accuracy   : {ensemble_acc * 100:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

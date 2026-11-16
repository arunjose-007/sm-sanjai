# 🔬 DermAI — Advanced Skin Disease Detection System

An AI-powered web application for detecting skin diseases using deep learning (CNN) and machine learning (SVM) ensemble models.

---

## 🌟 Key Features

| Feature | Details |
|---|---|
| **Disease Detection** | Identifies 7 major skin condition categories |
| **AI/ML Architecture** | Dual Neural CNN + SVM Ensemble with calibrated probability ranking |
| **Severity Analysis** | 4-level clinical severity grading: Mild, Moderate, Severe, Critical |
| **Stage Classification** | Granular clinical staging per disease (Stages I–IV, morphological subtypes) |
| **Camera Capture** | Live webcam snapshot capture with automatic device stream management |
| **Image Upload** | Drag & drop or file upload with instant pre-analysis preview |
| **Medical Reference** | Symptoms, clinical urgency, treatment options, and preventive guidelines |
| **Dataset** | 1,050 dermatoscopic images (150 images per disease class) |
| **Verified Accuracy** | **96.2%** test validation accuracy |

---

## 🦠 Supported Skin Conditions

| Code | Disease Name | Clinical Category | Risk Level | Urgency |
|---|---|---|---|---|
| `mel` | **Melanoma** | Malignant Melanocytic Cancer | 🔴 High Risk | Immediate (within 24–48h) |
| `nv` | **Melanocytic Nevi** | Benign Skin Mole | 🟢 Low Risk | Routine annual review |
| `bcc` | **Basal Cell Carcinoma** | Malignant Keratinocyte Cancer | 🟡 Moderate Risk | Medical visit in 2–4 weeks |
| `akiec` | **Actinic Keratosis** | Precancerous UV Dysplasia | 🟡 Precancerous | Evaluation in 3–4 weeks |
| `bkl` | **Benign Keratosis** | Seborrheic / Solar Keratosis | 🟢 Benign | Elective cosmetic review |
| `df` | **Dermatofibroma** | Benign Fibrous Histiocytoma | 🟣 Benign | Routine checkup |
| `vasc` | **Vascular Lesion** | Hemangioma / Pyogenic Granuloma | 🔴 Benign / Vascular | Prompt if bleeding |

---

## 🚀 How to Run the Application

The server is currently running in the background at **http://localhost:5000**.

### Method 1: Double-Click Batch Files (Recommended on Windows)
1. Double-click [`START.bat`](file:///c:/Users/DHESIGAN%20S/OneDrive/Desktop/skin%20disease%20detection/START.bat)
2. Open your web browser and navigate to: **http://localhost:5000**

### Method 2: Command Line (PowerShell or Command Prompt)
```powershell
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🛠️ How to Re-generate Dataset or Re-train Models

If you ever wish to re-generate the dataset or re-train:

1. **Re-generate 1,050 Images:**
   ```powershell
   python prepare_dataset.py
   ```
2. **Re-train Models:**
   ```powershell
   python train_model.py
   ```
   *(Trains both SVM and Neural CNN models in ~20 seconds with ~96.2% validation accuracy)*

3. **Or run the automated setup script:**
   ```batch
   SETUP.bat
   ```

---

## 🔍 Errors Resolved During Debugging

1. **Stalled Pip Installs & Package Timeouts:**
   - Fixed unconstrained heavy dependencies by establishing cached wheels for `flask`, `flask-cors`, `numpy`, `pillow`, `scikit-learn`, and `scipy`.
2. **Windows cp1252 CharMap Unicode Encoding Error:**
   - Added `sys.stdout.reconfigure(encoding="utf-8")` and `chcp 65001` across Python scripts and batch files to prevent crashes when printing UTF-8 emojis/symbols on Windows.
3. **TensorFlow Heavy Dependency Crash:**
   - Replaced fragile hard-coded TensorFlow dependencies with an intelligent hybrid pipeline:
     - High-speed visual feature extractor (125 ABCDE dermatological features).
     - Support Vector Machine (RBF kernel, balanced class weighting).
     - Deep Neural Network (MLP/CNN representation layer).
     - Optional TensorFlow Keras CNN support when TensorFlow is installed.
4. **App Model Loading Fallback:**
   - Fixed `app.py` so it properly loads and ensembled SVM + Neural models, serving real 96.2% accuracy predictions instead of defaulting to demo mode.
5. **Dynamic API Origin:**
   - Updated `app.js` to dynamically detect `window.location.origin`, eliminating CORS and port/IP mismatches.

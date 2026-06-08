#!/usr/bin/env python3
"""
Task 6 — Nationality Detection GUI
Run this script on your local machine (not in Colab).
Requirements: pip install tensorflow opencv-python PyQt5

Usage:
    python task6_gui.py

Set MODEL_DIR to the folder where your .h5 files are saved.
"""

import sys, os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QScrollArea, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import tensorflow as tf

# ── CONFIG — update this path ──────────────────────────────────────────────────
MODEL_DIR = r"C:/InternshipModels/Task6_Nationality_Detection/Models"
# ──────────────────────────────────────────────────────────────────────────────

NATIONALITY_CLASSES = ["American", "African", "Indian", "Others"]
EMOTION_CLASSES     = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
AGE_CLASSES         = ["Child", "Teen", "Adult", "Senior"]
COLOUR_CLASSES      = ["Red", "Blue", "Green", "Yellow", "White", "Black", "Orange", "Purple"]

NATIONALITY_FLAG = {
    "Indian": "[IN]", "American": "[US]", "African": "[AF]", "Others": "[--]"
}


def load_models():
    models = {}
    for name, fname in [
        ("nationality", "nationality_cnn.h5"),
        ("emotion",     "emotion_cnn.h5"),
        ("age",         "age_cnn.h5"),
        ("colour",      "dress_colour_cnn.h5"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)
            print(f"Loaded: {fname}")
        else:
            print(f"WARNING: {fname} not found at {path}")
    return models


def preprocess(img_bgr, size):
    img = cv2.resize(img_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return img[np.newaxis]


def run_pipeline(img_bgr, models):
    results = []

    # Nationality
    pred = models["nationality"].predict(preprocess(img_bgr, (128,128)), verbose=0)[0]
    nat  = NATIONALITY_CLASSES[np.argmax(pred)]
    results.append(("Nationality", nat, float(np.max(pred))))

    # Emotion (always)
    pred = models["emotion"].predict(preprocess(img_bgr, (64,64)), verbose=0)[0]
    results.append(("Emotion", EMOTION_CLASSES[np.argmax(pred)], float(np.max(pred))))

    # Age — Indian or American only
    if nat in ("Indian", "American") and "age" in models:
        pred = models["age"].predict(preprocess(img_bgr, (128,128)), verbose=0)[0]
        results.append(("Age", AGE_CLASSES[np.argmax(pred)], float(np.max(pred))))

    # Dress Colour — Indian or African only
    if nat in ("Indian", "African") and "colour" in models:
        pred = models["colour"].predict(preprocess(img_bgr, (64,64)), verbose=0)[0]
        results.append(("Dress Colour", COLOUR_CLASSES[np.argmax(pred)], float(np.max(pred))))

    return results, nat


class InferenceThread(QThread):
    finished = pyqtSignal(list, str)
    error    = pyqtSignal(str)

    def __init__(self, image_path, models):
        super().__init__()
        self.image_path = image_path
        self.models = models

    def run(self):
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                self.error.emit("Cannot read image.")
                return
            results, nationality = run_pipeline(img, self.models)
            self.finished.emit(results, nationality)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.setWindowTitle("Task 6 — Nationality Detection")
        self.setMinimumSize(860, 600)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # Title
        title = QLabel("Nationality Detection System")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("Upload an image to detect Nationality, Emotion, Age, and Dress Colour")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 11px;")
        root.addWidget(subtitle)

        # Content row: image | results
        content = QHBoxLayout()
        root.addLayout(content)

        # Image panel
        img_frame = QFrame()
        img_frame.setFrameShape(QFrame.StyledPanel)
        img_frame.setStyleSheet("background:#1a1a2e; border-radius:8px;")
        img_layout = QVBoxLayout(img_frame)
        self.img_label = QLabel("No image loaded")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(380, 380)
        self.img_label.setStyleSheet("color:#aaa; font-size:13px;")
        img_layout.addWidget(self.img_label)
        content.addWidget(img_frame, 55)

        # Results panel
        res_frame = QFrame()
        res_frame.setFrameShape(QFrame.StyledPanel)
        res_frame.setStyleSheet("background:#0f3460; border-radius:8px;")
        res_layout = QVBoxLayout(res_frame)

        res_title = QLabel("Results")
        res_title.setFont(QFont("Arial", 13, QFont.Bold))
        res_title.setStyleSheet("color:white;")
        res_title.setAlignment(Qt.AlignCenter)
        res_layout.addWidget(res_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none;")
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.results_container)
        res_layout.addWidget(scroll)

        self._add_placeholder_result()
        content.addWidget(res_frame, 45)

        # Buttons
        btn_row = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setMinimumHeight(42)
        self.upload_btn.setStyleSheet(
            "QPushButton{background:#16213e;color:white;border-radius:6px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#0f3460;}"
        )
        self.upload_btn.clicked.connect(self._open_file)

        self.analyze_btn = QPushButton("Analyze Image")
        self.analyze_btn.setMinimumHeight(42)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(
            "QPushButton{background:#e94560;color:white;border-radius:6px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#c73652;}"
            "QPushButton:disabled{background:#555;}"
        )
        self.analyze_btn.clicked.connect(self._analyze)

        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.analyze_btn)
        root.addLayout(btn_row)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color:#888; font-size:10px;")
        root.addWidget(self.status_label)

        self.setStyleSheet("background:#16213e; color:white;")

    def _add_placeholder_result(self):
        lbl = QLabel("Upload an image to see results here.")
        lbl.setStyleSheet("color:#aaa; font-style:italic;")
        lbl.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(lbl)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path: return
        self.current_path = path
        pixmap = QPixmap(path).scaled(370, 370, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText(f"Image loaded: {os.path.basename(path)}")

    def _analyze(self):
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("Analyzing...")
        self._clear_results()

        self.thread = InferenceThread(self.current_path, self.models)
        self.thread.finished.connect(self._show_results)
        self.thread.error.connect(self._show_error)
        self.thread.start()

    def _clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def _show_results(self, results, nationality):
        flag = NATIONALITY_FLAG.get(nationality, "[--]")

        COLOURS = {
            "Nationality":   "#2196F3",
            "Emotion":       "#4CAF50",
            "Age":           "#FF9800",
            "Dress Colour":  "#9C27B0",
        }

        for label_name, value, confidence in results:
            frame = QFrame()
            frame.setStyleSheet(f"background:#1a1a3e; border-radius:6px; margin:2px;")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(10, 8, 10, 8)

            display = f"{flag} {value}" if label_name == "Nationality" else value
            colour  = COLOURS.get(label_name, "white")

            top = QLabel(f"<b style='color:{colour};font-size:11px;'>{label_name}</b>")
            top.setTextFormat(Qt.RichText)

            val_lbl = QLabel(f"<span style='font-size:20px; font-weight:bold;'>{display}</span>")
            val_lbl.setTextFormat(Qt.RichText)

            conf_lbl = QLabel(f"Confidence: {confidence:.2%}")
            conf_lbl.setStyleSheet("color:#aaa; font-size:10px;")

            layout.addWidget(top)
            layout.addWidget(val_lbl)
            layout.addWidget(conf_lbl)
            self.results_layout.addWidget(frame)

        self.analyze_btn.setEnabled(True)
        self.status_label.setText(f"Analysis complete — {nationality} pipeline applied.")

    def _show_error(self, msg):
        lbl = QLabel(f"Error: {msg}")
        lbl.setStyleSheet("color:red;")
        self.results_layout.addWidget(lbl)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Error during analysis.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    print("Loading models...")
    models = load_models()
    window = MainWindow(models)
    window.show()
    sys.exit(app.exec_())
import sys
import torch
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from test_model_core import load_model_and_predict  # Model kodlarını ayıracağız

class CancerPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kanser Tahmin Arayüzü")
        self.setGeometry(100, 100, 400, 200)
        self.init_ui()

    def init_ui(self):
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("TEXT (biyopsi yeri vs)")

        self.gene_input = QLineEdit(self)
        self.gene_input.setPlaceholderText("Gene (örn: 'Male_65')")

        self.variation_input = QLineEdit(self)
        self.variation_input.setPlaceholderText("Variation (örn: 'Caucasian_FamilyHistory')")

        self.result_label = QLabel("Sonuç: ", self)

        predict_button = QPushButton("Tahmin Et", self)
        predict_button.clicked.connect(self.predict_cancer)

        layout = QVBoxLayout()
        layout.addWidget(self.text_input)
        layout.addWidget(self.gene_input)
        layout.addWidget(self.variation_input)
        layout.addWidget(predict_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict_cancer(self):
        text = self.text_input.text()
        gene = self.gene_input.text()
        variation = self.variation_input.text()

        prediction, confidence = load_model_and_predict(text)
        self.result_label.setText(f"Tahmin: {prediction}\nGüven: %{confidence:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CancerPredictionApp()
    window.show()
    sys.exit(app.exec_())

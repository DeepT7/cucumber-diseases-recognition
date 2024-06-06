import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from PySide6.QtGui import QPixmap
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import cv2
import os
import matplotlib.pyplot as plt

classes = ['Anthracnose', 'Bacterial Wilt', 'Belly Rot', 'Downy Mildew', 'Fresh Cucumber', 'Fresh Leaf', 'Gummy Stem Blight', 'Pythium Fruit Rot']
model = load_model('./trained model/trained_model_cucu.h5', compile = False)
model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Placeholder function for your deep learning model processing
def process_image(input_image_path):
    # Perform your model's processing here
    # Load the image and process
    img = image.load_img(input_image_path, target_size = (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = x/255.0
    
    # Make a prediction
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    y_pred = classes[predicted_class]

    output_image_path = os.path.join('./images', y_pred)
    
    return output_image_path

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up the main layout
        main_layout = QVBoxLayout()

        # Set up the image layout to hold the input and output images side by side
        image_layout = QHBoxLayout()
        
        # Input image label
        self.inputImageLabel = QLabel("No image selected")
        self.inputImageLabel.setFixedSize(400, 400)
        self.inputImageLabel.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.inputImageLabel)

        # Output image label
        self.outputImageLabel = QLabel("Output will be displayed here")
        self.outputImageLabel.setFixedSize(400, 400)
        self.outputImageLabel.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.outputImageLabel)

        # Add the image layout to the main layout
        main_layout.addLayout(image_layout)

        # Select button
        self.selectButton = QPushButton("Select Image")
        self.selectButton.clicked.connect(self.openFileNameDialog)
        main_layout.addWidget(self.selectButton)

        # Process button
        self.processButton = QPushButton("Process")
        self.processButton.clicked.connect(self.processImage)
        main_layout.addWidget(self.processButton)

        self.setLayout(main_layout)
        self.setWindowTitle("Image Processor")
        self.setGeometry(100, 100, 850, 500)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if fileName:
            self.inputImagePath = fileName
            self.inputImageLabel.setPixmap(QPixmap(fileName).scaled(400, 400))

    def processImage(self):
        if hasattr(self, 'inputImagePath'):
            outputImagePath = process_image(self.inputImagePath)
            self.outputImageLabel.setPixmap(QPixmap(outputImagePath).scaled(400, 400))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessorApp()
    ex.show()
    sys.exit(app.exec())

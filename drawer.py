import sys
import pickle
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt
from test import NN

class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280) 
        self.drawing = False
        self.last_point = None
        self.image = QPixmap(280, 280)
        self.image.fill(Qt.white)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.image)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            
    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        
    def get_digit_data(self):
        # Convert the QPixmap to a 28x28 numpy array
        image = self.image.toImage()
        buffer = QPixmap.fromImage(image).toImage()
        buffer = buffer.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Turn the QPixmap into a 28x28 numpy array
        ptr = buffer.bits()
        ptr.setsize(buffer.byteCount())
        arr = np.array(ptr).reshape(28, 28, 4)  # RGBA
        
        # To grayscale
        arr = arr[:,:,0]
        arr = 255 - arr 
        
        # Normalize
        arr = arr.astype('float32') / 255.0
        return arr.reshape(1, 784)  # Flatttten



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HWR by JJJAMES')
        self.setFixedSize(300, 400)
        
        # Load the model
        with open('mnistHandwrite.pkl', 'rb') as file:
            self.model = pickle.load(file)
            
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.drawing_widget = DrawingWidget()
        layout.addWidget(self.drawing_widget)
        self.result_label = QLabel('Write down a digit and click the button')
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)
        
        predict_btn = QPushButton('Predict')
        predict_btn.clicked.connect(self.predict_digit)
        layout.addWidget(predict_btn)
        
        clear_btn = QPushButton('Clear')
        clear_btn.clicked.connect(self.drawing_widget.clear)
        layout.addWidget(clear_btn)
        
    def predict_digit(self):
        digit_data = self.drawing_widget.get_digit_data()
        
        prediction = self.model.predict(digit_data)
        self.result_label.setText(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

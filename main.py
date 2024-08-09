from PyQt6.QtWidgets import QApplication
import signal # lets control-C actually close the app
from main_window import MainWindow
import sys

app = QApplication([])
window = MainWindow()
window.show() # Windows are hidden by default
signal.signal(signal.SIGINT, signal.SIG_DFL) # this lets control-C actually close the app
app.exec() # Start the event loop

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from main_window import MainWindow
class AppBootstrap:
    def __init__(self, argv):
        self.app = QApplication(argv)
        self.app.setStyle("Fusion")
        self.app.setWindowIcon(QIcon(str(Path(__file__).parent / "icon.png")))
    def run(self):
        win = MainWindow()
        win.show()
        return self.app.exec_()
if __name__ == "__main__":
    sys.exit(AppBootstrap(sys.argv).run())
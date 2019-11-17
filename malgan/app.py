from PyQt5.QtWidgets import QDialog

from malgan.dialog import malganDialog


class AppWindow(QDialog):
    def __init__(self, main_fct):
        super().__init__()
        self.ui = malganDialog(main_fct)
        self.ui.setupUi(self)
        self.show()

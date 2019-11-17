from PyQt5.QtWidgets import QDialog

from malgan.dialog import MyDialog


class AppWindow(QDialog):
    def __init__(self, main_fct):
        super().__init__()
        self.ui = MyDialog(main_fct)
        self.ui.setupUi(self)
        self.show()

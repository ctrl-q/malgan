from PyQt5.QtWidgets import (QWidget)

from malgan.QTDialog import QTDialog


class QTApp(QWidget):
    from PyQt5.QtWidgets import QDialog

    class AppWindow(QDialog):
        def __init__(self, main_fct):
            super().__init__()
            self.ui = QTDialog(main_fct=main_fct)
            self.ui.setupUi(self)
            self.show()

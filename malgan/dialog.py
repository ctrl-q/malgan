import multiprocessing

from PyQt5 import QtCore, QtGui, QtWidgets


class malganDialog(object):
    def __init__(self, main_fct):
        self.main_fct = main_fct
        self.main_fct_process = multiprocessing.Process(target=self.main_fct)

        self.dialog = None

        super().__init__()

    def start_main_fct_thread(self):
        self.main_fct_process.start()

        self.plainTextEdit.appendPlainText("App started successfully.")

    def stop_main_fct_process(self):
        if self.main_fct_process.is_alive():
            self.main_fct_process.terminate()

        self.plainTextEdit.appendPlainText("App stopped succesfully.")
        self.plainTextEdit.appendPlainText("Now closing.")
        self.dialog.accept()

    def setupUi(self, dialog):
        self.dialog = dialog

        dialog.setObjectName("Dialog")
        dialog.resize(600, 500)
        font = QtGui.QFont()
        font.setFamily("Open Sans Light")
        dialog.setFont(font)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(dialog)
        self.plainTextEdit.setGeometry(QtCore.QRect(20, 380, 400, 100))
        self.plainTextEdit.setObjectName("plainTextEdit")

        self.button_start = QtWidgets.QPushButton(dialog)
        self.button_start.clicked.connect(self.start_main_fct_thread)
        self.button_start.setGeometry(QtCore.QRect(460, 400, 100, 25))
        self.button_start.setObjectName("pushButton")

        self.button_stop = QtWidgets.QPushButton(dialog)
        self.button_stop.clicked.connect(self.stop_main_fct_process)
        self.button_stop.setGeometry(QtCore.QRect(460, 430, 100, 25))
        self.button_stop.setObjectName("pushButton_2")

        self.listWidget = QtWidgets.QListWidget(dialog)
        self.listWidget.setGeometry(QtCore.QRect(20, 40, 561, 192))
        self.listWidget.setObjectName("listWidget")

        self.button_add_to_api_list = QtWidgets.QPushButton(dialog)
        self.button_add_to_api_list.setGeometry(QtCore.QRect(460, 240, 91, 81))
        self.button_add_to_api_list.setObjectName("pushButton_3")

        self.comboBox = QtWidgets.QComboBox(dialog)
        self.comboBox.setGeometry(QtCore.QRect(20, 270, 400, 25))

        font = QtGui.QFont()
        font.setFamily("Open Sans Light")

        self.comboBox.setFont(font)
        self.comboBox.setCurrentText("")
        self.comboBox.setObjectName("comboBox")

        self.label_list_of_apis = QtWidgets.QLabel(dialog)
        self.label_list_of_apis.setGeometry(QtCore.QRect(20, 250, 61, 16))
        self.label_list_of_apis.setObjectName("label")

        self.label_chosen_apis = QtWidgets.QLabel(dialog)
        self.label_chosen_apis.setGeometry(QtCore.QRect(20, 20, 61, 16))
        self.label_chosen_apis.setObjectName("label_2")

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("Dialog", "malGANs"))
        self.button_start.setText(_translate("Dialog", "Start"))
        self.button_stop.setText(_translate("Dialog", "Stop"))
        self.button_add_to_api_list.setText(_translate("Dialog", "Add to API list."))
        self.label_list_of_apis.setText(_translate("Dialog", "List of apis."))
        self.label_chosen_apis.setText(_translate("Dialog", "Chosen apis."))

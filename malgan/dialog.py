import multiprocessing

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLineEdit

api_func_in_string = ['copy_file', 'create_directory', 'delete_file', 'download_file',
                      'get_computer_name', 'get_file_info', 'get_free_disk_space',
                      'get_short_path_name', 'get_system_directory', 'get_temp_dir', 'get_time',
                      'get_username', 'read_file', 'remove_directory', 'set_file_time',
                      'terminate_process', 'write_console', 'write_file']

api_func_in_string.sort()


class malganDialog(object):
    def __init__(self, main_fct):
        self.main_fct = main_fct

        self.final_list_of_apis = None

        super().__init__()

    def start_main_fct_thread(self):
        if self.main_fct_process.is_alive():
            self.main_fct_process.terminate()

        self.main_fct_process.start()

        self.consoleListWidget.addItem("App started successfully.")

    def stop_main_fct_process(self):
        if self.main_fct_process.is_alive():
            self.main_fct_process.terminate()

        self.consoleListWidget.addItem("App stopped succesfully.")
        self.consoleListWidget.addItem("Now closing.")

        self.dialog.accept()

    def add_to_apis_list(self):
        selected_api = self.comboBox.currentText()

        list_of_apis = self.getListOfApis()

        if selected_api in list_of_apis:
            output_message = selected_api + " is already selected!"
            self.consoleListWidget.addItem(output_message)
        else:
            self.listWidget.addItem(selected_api)

        self.updateStartButton()

    def clear_apis_list(self):
        self.listWidget.clear()

        self.consoleListWidget.addItem("Cleared apis list.")

    def getListOfApis(self):
        list = []
        for i in range(self.listWidget.count()):
            elem = self.listWidget.item(i).text()
            list.append(elem)

        return list

    def updateStartButton(self):
        self.final_list_of_apis = self.getListOfApis()

        try:
            z = int(self.z_textbox.text())
            batch_size = int(self.batch_size_textbox.text())
            num_epochs = int(self.num_epochs_textbox.text())
            hidden_size_gen = int(self.hidden_size_gen_textbox.text())
            hidden_size_dis = int(self.hidden_size_dis_textbox.text())

            self.main_fct_process = multiprocessing.Process(target=self.main_fct, args=(z, batch_size, num_epochs, hidden_size_gen, hidden_size_dis, self.final_list_of_apis))

            self.rewriteStartButton()
        except:
            self.consoleListWidget.addItem("Invalid parameters.")

    def rewriteStartButton(self):
        self.button_start = QtWidgets.QPushButton(self.dialog)
        self.button_start.clicked.connect(self.start_main_fct_thread)
        self.button_start.setGeometry(QtCore.QRect(460, 400, 100, 25))
        self.button_start.setObjectName("start")

    def setupUi(self, dialog):
        self.dialog = dialog

        dialog.setObjectName("Dialog")
        dialog.resize(600, 500)
        font = QtGui.QFont()
        font.setFamily("Open Sans Light")
        dialog.setFont(font)
        self.consoleListWidget = QtWidgets.QListWidget(dialog)
        self.consoleListWidget.setGeometry(QtCore.QRect(20, 380, 400, 100))
        self.consoleListWidget.setObjectName("console")

        self.console_label = QtWidgets.QLabel(dialog)
        self.console_label.setGeometry(QtCore.QRect(20, 313, 400, 100))
        self.console_label.setObjectName("console_label")

        self.button_start = QtWidgets.QPushButton(self.dialog)
        self.button_start.clicked.connect(self.start_main_fct_thread)
        self.button_start.setGeometry(QtCore.QRect(460, 400, 100, 25))
        self.button_start.setObjectName("start")

        self.button_stop = QtWidgets.QPushButton(dialog)
        self.button_stop.clicked.connect(self.stop_main_fct_process)
        self.button_stop.setGeometry(QtCore.QRect(460, 430, 100, 25))
        self.button_stop.setObjectName("stop")

        self.listWidget = QtWidgets.QListWidget(dialog)
        self.listWidget.setGeometry(QtCore.QRect(20, 40, 400, 192))
        self.listWidget.setObjectName("list_apis")

        self.z_label = QtWidgets.QLabel(dialog)
        self.z_label.setGeometry(QtCore.QRect(460, 36, 91, 15))
        self.z_label.setObjectName("z_label")
        self.z_textbox = QLineEdit(dialog)
        self.z_textbox.setGeometry(QtCore.QRect(460, 51, 91, 20))
        self.z_textbox.setObjectName("z_textbox")

        self.batch_size_label = QtWidgets.QLabel(dialog)
        self.batch_size_label.setGeometry(QtCore.QRect(460, 75, 130, 15))
        self.batch_size_label.setObjectName("batch_size_label")
        self.batch_size_textbox = QLineEdit(dialog)
        self.batch_size_textbox.setGeometry(QtCore.QRect(460, 90, 91, 20))
        self.batch_size_textbox.setObjectName("batch_size_textbox")

        self.num_epochs_label = QtWidgets.QLabel(dialog)
        self.num_epochs_label.setGeometry(QtCore.QRect(460, 114, 130, 15))
        self.num_epochs_label.setObjectName("num_epochs_label")
        self.num_epochs_textbox = QLineEdit(dialog)
        self.num_epochs_textbox.setGeometry(QtCore.QRect(460, 129, 91, 20))
        self.num_epochs_textbox.setObjectName("num_epochs_textbox")

        self.hidden_size_gen_label = QtWidgets.QLabel(dialog)
        self.hidden_size_gen_label.setGeometry(QtCore.QRect(460, 153, 130, 15))
        self.hidden_size_gen_label.setObjectName("hidden_size_gen_label")
        self.hidden_size_gen_textbox = QLineEdit(dialog)
        self.hidden_size_gen_textbox.setGeometry(QtCore.QRect(460, 168, 91, 20))
        self.hidden_size_gen_textbox.setObjectName("hidden_size_gen_textbox")

        self.hidden_size_dis_label = QtWidgets.QLabel(dialog)
        self.hidden_size_dis_label.setGeometry(QtCore.QRect(460, 192, 130, 15))
        self.hidden_size_dis_label.setObjectName("hidden_size_dis_label")
        self.hidden_size_dis_textbox = QLineEdit(dialog)
        self.hidden_size_dis_textbox.setGeometry(QtCore.QRect(460, 207, 91, 20))
        self.hidden_size_dis_textbox.setObjectName("hidden_size_dis_textbox")

        self.comboBox = QtWidgets.QComboBox(dialog)
        self.comboBox.setGeometry(QtCore.QRect(20, 270, 400, 25))

        font = QtGui.QFont()
        font.setFamily("Open Sans Light")

        self.comboBox.setFont(font)
        self.comboBox.setCurrentText("")
        self.comboBox.clear()
        self.comboBox.addItems(api_func_in_string)
        self.comboBox.setObjectName("select_box")

        self.button_add_to_api_list = QtWidgets.QPushButton(dialog)
        self.button_add_to_api_list.setGeometry(QtCore.QRect(460, 240, 91, 81))
        self.button_add_to_api_list.setObjectName("add_to_api_list")
        self.button_add_to_api_list.clicked.connect(self.add_to_apis_list)

        self.label_list_of_apis = QtWidgets.QLabel(dialog)
        self.label_list_of_apis.setGeometry(QtCore.QRect(20, 245, 61, 16))
        self.label_list_of_apis.setObjectName("list_of_apis")

        self.label_chosen_apis = QtWidgets.QLabel(dialog)
        self.label_chosen_apis.setGeometry(QtCore.QRect(20, 15, 61, 16))
        self.label_chosen_apis.setObjectName("chosen_apis")

        self.button_clear = QtWidgets.QPushButton(dialog)
        self.button_clear.clicked.connect(self.clear_apis_list)
        self.button_clear.setGeometry(QtCore.QRect(329, 10, 91, 25))
        self.button_clear.setObjectName("clear")

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

        self.dialog = dialog

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("Dialog", "malGANs"))
        self.button_clear.setText(_translate("Dialog", "Clear"))
        self.button_start.setText(_translate("Dialog", "Start"))
        self.button_stop.setText(_translate("Dialog", "Stop"))
        self.button_add_to_api_list.setText(_translate("Dialog", "Add to API list."))
        self.label_list_of_apis.setText(_translate("Dialog", "List of apis."))
        self.label_chosen_apis.setText(_translate("Dialog", "Chosen apis."))
        self.z_label.setText(_translate("Dialog", "Z:"))
        self.batch_size_label.setText(_translate("Dialog", "Batch size:"))
        self.num_epochs_label.setText(_translate("Dialog", "Num epochs:"))
        self.console_label.setText(_translate("Dialog", "Console."))
        self.hidden_size_gen_label.setText(_translate("Dialog", "Hidden size generator."))
        self.hidden_size_dis_label.setText(_translate("Dialog", "Hidden size discriminator."))

        self.z_textbox.setText("10")
        self.batch_size_textbox.setText("32")
        self.num_epochs_textbox.setText("100")
        self.hidden_size_gen_textbox.setText("256")
        self.hidden_size_dis_textbox.setText("256")

        z = int(self.z_textbox.text())
        batch_size = int(self.batch_size_textbox.text())
        num_epochs = int(self.num_epochs_textbox.text())
        hidden_size_gen = int(self.hidden_size_gen_textbox.text())
        hidden_size_dis = int(self.hidden_size_dis_textbox.text())
        self.main_fct_process = multiprocessing.Process(target=self.main_fct, args=(z, batch_size, num_epochs, hidden_size_gen, hidden_size_dis, self.final_list_of_apis))

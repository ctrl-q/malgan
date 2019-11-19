import multiprocessing

from PyQt5 import QtCore, QtGui, QtWidgets

from script.genscript import *

api_map = {  # maps array indices to code requiring specific Windows APIs
    "terminate_process": terminate_process,
    "get_file_info": get_file_info,
    "write_console": write_console,
    "get_short_path_name": get_short_path_name,
    "get_temp_dir": get_temp_dir,
    "get_file_info": get_file_info,
    "get_file_info": get_file_info,
    "get_file_info": get_file_info,
    "create_directory": create_directory,
    "get_system_directory": get_system_directory,
    "get_file_info": get_file_info,
    "get_file_info": get_file_info,
    "get_time": get_time,
    "delete_file": delete_file,
    "get_file_info": get_file_info,
    "write_file": write_file,
    "read_file": read_file,
    "get_file_info": get_file_info,
    "write_file": write_file,
    "get_computer_name": get_computer_name,
    "get_file_info": get_file_info,
    "read_file": read_file,
    "read_file": read_file,
    "get_system_directory": get_system_directory,
    "get_system_directory": get_system_directory,
    "get_file_info": get_file_info,
    "get_computer_name": get_computer_name,
    "get_time": get_time,
    "get_file_info": get_file_info,
    "copy_file": copy_file,
    "write_console": write_console,
    "get_system_directory": get_system_directory,
    "get_username": get_username,
    "get_file_info": get_file_info,
    "get_username": get_username,
    "set_file_time": set_file_time,
    "copy_file": copy_file,
    "copy_file": copy_file,
    "get_username": get_username,
    "get_username": get_username,
    "remove_directory": remove_directory,
    "get_free_disk_space": get_free_disk_space,
    "remove_directory": remove_directory,
    "get_system_directory": get_system_directory,
    "download_file": download_file,
    "get_free_disk_space": get_free_disk_space,
    "create_directory": create_directory,
    "download_file": download_file,
    "delete_file": delete_file,
    "get_file_info": get_file_info
}


class malganDialog(object):
    def __init__(self, main_fct):
        self.main_fct = main_fct
        self.main_fct_process = multiprocessing.Process(target=self.main_fct)

        self.dialog = None

        super().__init__()

    def clear_apis_list(self):
        self.listWidget.clear()

        self.consoleListWidget.addItem("Cleared apis list.")

    def start_main_fct_thread(self):
        self.main_fct_process.start()

        self.consoleListWidget.addItem("App started successfully.")

    def stop_main_fct_process(self):
        if self.main_fct_process.is_alive():
            self.main_fct_process.terminate()

        self.consoleListWidget.addItem("App stopped succesfully.")
        self.consoleListWidget.addItem("Now closing.")
        
        self.dialog.accept()

    def getListOfApis(self):
        list = []
        for i in range(self.listWidget.count()):
            elem = self.listWidget.item(i).text()
            list.append(elem)

        return list

    def add_to_apis_list(self):
        selected_api = self.comboBox.currentText()

        list_of_apis = self.getListOfApis()

        if selected_api in list_of_apis:
            output_message = selected_api + " is already selected!"
            self.consoleListWidget.addItem(output_message)
        else:
            self.listWidget.addItem(selected_api)

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

        self.button_start = QtWidgets.QPushButton(dialog)
        self.button_start.clicked.connect(self.start_main_fct_thread)
        self.button_start.setGeometry(QtCore.QRect(460, 400, 100, 25))
        self.button_start.setObjectName("start")

        self.button_stop = QtWidgets.QPushButton(dialog)
        self.button_stop.clicked.connect(self.stop_main_fct_process)
        self.button_stop.setGeometry(QtCore.QRect(460, 430, 100, 25))
        self.button_stop.setObjectName("stop")

        self.listWidget = QtWidgets.QListWidget(dialog)
        self.listWidget.setGeometry(QtCore.QRect(20, 40, 561, 192))
        self.listWidget.setObjectName("list_apis")

        self.comboBox = QtWidgets.QComboBox(dialog)
        self.comboBox.setGeometry(QtCore.QRect(20, 270, 400, 25))

        font = QtGui.QFont()
        font.setFamily("Open Sans Light")

        self.comboBox.setFont(font)
        self.comboBox.setCurrentText("")
        list_of_apis = api_map.keys()
        self.comboBox.clear()
        self.comboBox.addItems(list_of_apis)
        self.comboBox.setObjectName("select_box")

        self.button_add_to_api_list = QtWidgets.QPushButton(dialog)
        self.button_add_to_api_list.setGeometry(QtCore.QRect(460, 240, 91, 81))
        self.button_add_to_api_list.setObjectName("add_to_api_list")
        self.button_add_to_api_list.clicked.connect(self.add_to_apis_list)

        self.label_list_of_apis = QtWidgets.QLabel(dialog)
        self.label_list_of_apis.setGeometry(QtCore.QRect(20, 250, 61, 16))
        self.label_list_of_apis.setObjectName("label")

        self.label_chosen_apis = QtWidgets.QLabel(dialog)
        self.label_chosen_apis.setGeometry(QtCore.QRect(20, 20, 61, 16))
        self.label_chosen_apis.setObjectName("label_2")

        self.button_clear = QtWidgets.QPushButton(dialog)
        self.button_clear.clicked.connect(self.clear_apis_list)
        self.button_clear.setGeometry(QtCore.QRect(460, 10, 91, 25))
        self.button_clear.setObjectName("clear")

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("Dialog", "malGANs"))
        self.button_clear.setText(_translate("Dialog", "Clear"))
        self.button_start.setText(_translate("Dialog", "Start"))
        self.button_stop.setText(_translate("Dialog", "Stop"))
        self.button_add_to_api_list.setText(_translate("Dialog", "Add to API list."))
        self.label_list_of_apis.setText(_translate("Dialog", "List of apis."))
        self.label_chosen_apis.setText(_translate("Dialog", "Chosen apis."))

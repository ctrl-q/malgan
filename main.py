import pickle
import sys
from pathlib import Path
from typing import Union

import torch
from PyQt5.QtWidgets import QApplication
from torch import nn

from malgan import MalGAN, MalwareDataset, BlackBoxDetector
from malgan.app import AppWindow
from script import genscript
from script.genscript import *


def _configure_activation_function(name_of_activation_func: str) -> nn.Module:
    name_of_activation_func = name_of_activation_func.lower()  # Make case insensitive

    act_funcs = [("relu", nn.ReLU), ("elu", nn.ELU), ("leakyrelu", nn.LeakyReLU), ("tanh", nn.Tanh),
                 ("sigmoid", nn.Sigmoid)]

    for func_name, module in act_funcs:
        if name_of_activation_func == func_name.lower():
            return module
    raise ValueError("Unknown activation function: \"%s\"" % name_of_activation_func)


def load_dataset(file_path: Union[str, Path], labels: int) -> MalwareDataset:
    file_extension = Path(file_path).suffix
    if file_extension in {".npy", ".npz"}:
        data = np.load(file_path)
    elif file_extension in {".pt", ".pth"}:
        data = torch.load(str(file_path))
    elif file_extension == ".pk":
        with open(str(file_path), "rb") as f_in:
            data = pickle.load(f_in)
    elif file_extension == ".csv":
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    else:
        raise ValueError("Unknown file extension.  Cannot determine how to import")
    return MalwareDataset(data=data, classes=labels)


def main(z, batch_size, num_epochs, hidden_size_gen, hidden_size_dis, pretrained_model_path, list_of_apis_to_use=None):
    if list_of_apis_to_use is None:
        list_of_apis_to_use = []

    if (pretrained_model_path != ""):
        try:
            pretrained_model_path = Path(pretrained_model_path)
        except:
            print("Invalid path. Now training a new model.")
            pretrained_model_path = None
    else:
        pretrained_model_path = None

    mal_file = "data/trial_mal.npy"
    ben_file = "data/trial_ben.npy"

    activation = _configure_activation_function("LeakyReLU")
    detector = BlackBoxDetector.Type.get_classifier_from_name(BlackBoxDetector.Type.RandomForest.name)

    mal_file = Path(mal_file)
    ben_file = Path(ben_file)
    for (name, path) in (("malware", mal_file), ("benign", ben_file)):
        if path.exists():
            continue
        print(f"Unknown %s file \"%s\"" % (name, str(path)))
        sys.exit(1)

    MalGAN.MALWARE_BATCH_SIZE = batch_size

    malgan = MalGAN(load_dataset(mal_file, MalGAN.Label.Malware.value),
                    load_dataset(ben_file, MalGAN.Label.Benign.value),
                    dim_noise_vect=z,
                    hidden_layer_width_generator=hidden_size_gen,
                    hidden_layer_width_discriminator=hidden_size_dis,
                    generator_hidden=activation,
                    detector_type=detector,
                    path=pretrained_model_path)

    if pretrained_model_path is None:
        malgan.fit(num_epochs)

    malgan = malgan.cpu()

    binary_array, _ = malgan(next(iter(malgan._mal_data.test))[0][:1])
    array = binary_array.flatten().cpu().numpy()

    for func in list_of_apis_to_use:
        index = api_map_to_index[func]

        if index < 128:
            array[index] = 1

    genscript.generate(array)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AppWindow(main)
    w.show()
    sys.exit(app.exec_())

api_map_to_index = {
    "terminate_process": 0,
    "get_file_info": 1,
    "write_console": 8,
    "get_short_path_name": 11,
    "get_temp_dir": 20,
    "get_file_info": 22,
    "get_file_info": 23,
    "get_file_info": 28,
    "create_directory": 34,
    "get_system_directory": 37,
    "get_file_info": 42,
    "get_file_info": 44,
    "get_time": 49,
    "delete_file": 57,
    "get_file_info": 58,
    "write_file": 59,
    "read_file": 60,
    "get_file_info": 63,
    "write_file": 75,
    "get_computer_name": 83,
    "get_file_info": 84,
    "read_file": 91,
    "read_file": 92,
    "get_system_directory": 93,
    "get_system_directory": 95,
    "get_file_info": 113,
    "get_computer_name": 117,
    "get_time": 121,
    "get_file_info": 127,
    "copy_file": 125,
    "write_console": 134,
    "get_system_directory": 140,
    "get_username": 149,
    "get_file_info": 153,
    "get_username": 156,
    "set_file_time": 161,
    "copy_file": 162,
    "copy_file": 165,
    "get_username": 167,
    "get_username": 168,
    "remove_directory": 178,
    "get_free_disk_space": 183,
    "remove_directory": 184,
    "get_system_directory": 210,
    "download_file": 214,
    "get_free_disk_space": 215,
    "create_directory": 242,
    "download_file": 248,
    "delete_file": 254,
    "get_file_info": 259
}

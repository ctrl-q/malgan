import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("Z", help="Dimension of the latent vector", type=int, default=10)
    parser.add_argument("batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("num_epoch", help="Number of training epochs", type=int, default=100)

    message = "Data file contacting the %s feature vectors"
    for mode in ["malware", "benign"]:
        parser.add_argument(mode[:3] + "_file", help=message % mode, type=str, default="data/%s.npy" % mode)

    help_message = " ".join(["Dimension of the hidden layer(s) in the GENERATOR."
                             "Multiple layers should be space separated"])

    parser.add_argument("--gen-hidden-sizes", help=help_message, type=int,
                        default=[256, 256], nargs="+")

    help_message = " ".join(["Dimension of the hidden layer(s) in the DISCRIMINATOR."
                             "Multiple layers should be space separated"])

    parser.add_argument("--discrim-hidden-sizes", help=help_message, type=int,
                        default=[256, 256], nargs="+")

    help_message = " ".join(["Activation function for the generator and discriminator hidden",
                             "layer(s). Valid choices (case insensitive) are: \"ReLU\", \"ELU\",",
                             "\"LeakyReLU\", \"tanh\" and \"sigmoid\"."])

    parser.add_argument("--activation", help=help_message, type=str, default="LeakyReLU")

    help_message = " ".join(["Path to a pretrained .pth file.", "If None, a model will be trained"])

    parser.add_argument("--pth", help=help_message, type=Path)

    help_message = ["Learner algorithm used in the black box detector. Valid choices (case ",
                    "insensitive) include:"]

    names = BlackBoxDetector.Type.names()
    for i, type_name in enumerate(names):
        if i > 0 and len(names) > 2:
            help_message.append(",")

        if len(names) > 1 and i == len(names) - 1:
            help_message.append(" and")

        help_message.extend([" \"", type_name, "\""])
    help_message.append(".")
    parser.add_argument("--detector", help="".join(help_message), type=str,
                        default=BlackBoxDetector.Type.RandomForest.name)

    args = parser.parse_args()
    args.activation = _configure_activation_function(args.activation)
    args.detector = BlackBoxDetector.Type.get_classifier_from_name(args.detector)

    args.mal_file = Path(args.mal_file)
    args.ben_file = Path(args.ben_file)
    for (name, path) in (("malware", args.mal_file), ("benign", args.ben_file)):
        if path.exists():
            continue
        print(f"Unknown %s file \"%s\"" % (name, str(path)))
        sys.exit(1)
    return args


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
    else:
        raise ValueError("Unknown file extension.  Cannot determine how to import")
    return MalwareDataset(data=data, classes=labels)


def main(list_of_apis_to_use=None):
    if list_of_apis_to_use is None:
        list_of_apis_to_use = []

    args = parse_args()

    MalGAN.MALWARE_BATCH_SIZE = args.batch_size

    malgan = MalGAN(load_dataset(args.mal_file, MalGAN.Label.Malware.value),
                    load_dataset(args.ben_file, MalGAN.Label.Benign.value),
                    dim_noise_vect=args.Z,
                    hidden_layer_width_generator=args.gen_hidden_sizes,
                    hidden_layer_width_discriminator=args.discrim_hidden_sizes,
                    generator_hidden=args.activation,
                    detector_type=args.detector,
                    path=args.pth)
    if not args.pth:
        malgan.fit(args.num_epoch)

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

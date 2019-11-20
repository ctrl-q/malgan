import argparse
import pickle
import sys
from pathlib import Path
from typing import Union

import torch
from PyQt5.QtWidgets import QApplication
from torch import nn

from malgan import MalGAN, MalwareDataset, BlackBoxDetector, setup_logger
from malgan.app import AppWindow
from malgan.dialog import api_map_to_func
from script import genscript
from script.genscript import *


def parse_args() -> argparse.Namespace:
    r"""
    Parse the command line arguments

    :return: Parsed argument structure
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("Z", help="Dimension of the latent vector", type=int, default=10)
    parser.add_argument("batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("num_epoch", help="Number of training epochs", type=int, default=100)

    msg = "Data file contacting the %s feature vectors"
    for x in ["malware", "benign"]:
        parser.add_argument(x[:3] + "_file", help=msg % x, type=str, default="data/%s.npy" % x)

    parser.add_argument("-q", help="Quiet mode", action='store_true', default=False)

    help_msg = " ".join(["Dimension of the hidden layer(s) in the GENERATOR."
                         "Multiple layers should be space separated"])
    parser.add_argument("--gen-hidden-sizes", help=help_msg, type=int,
                        default=[256, 256], nargs="+")

    help_msg = " ".join(["Dimension of the hidden layer(s) in the DISCRIMINATOR."
                         "Multiple layers should be space separated"])
    parser.add_argument("--discrim-hidden-sizes", help=help_msg, type=int,
                        default=[256, 256], nargs="+")

    help_msg = " ".join(["Activation function for the generator and discriminator hidden",
                         "layer(s). Valid choices (case insensitive) are: \"ReLU\", \"ELU\",",
                         "\"LeakyReLU\", \"tanh\" and \"sigmoid\"."])
    parser.add_argument("--activation", help=help_msg, type=str, default="LeakyReLU")

    help_msg = " ".join(["Path to a pretrained .pth file.", "If None, a model will be trained"])
    parser.add_argument("--pth", help=help_msg, type=Path)

    help_msg = ["Learner algorithm used in the black box detector. Valid choices (case ",
                "insensitive) include:"]
    names = BlackBoxDetector.Type.names()
    for i, type_name in enumerate(names):
        if i > 0 and len(names) > 2:  # Need three options for a comma to make sense
            help_msg.append(",")
        if len(names) > 1 and i == len(names) - 1:  # And only makes sense if at least two options
            help_msg.append(" and")
        help_msg.extend([" \"", type_name, "\""])
    help_msg.append(".")
    parser.add_argument("--detector", help="".join(help_msg), type=str,
                        default=BlackBoxDetector.Type.RandomForest.name)

    help_msg = "Print the results to the console. Intended for slurm results analysis"
    parser.add_argument("--print-results", help=help_msg, action="store_true", default=False)

    args = parser.parse_args()
    # noinspection PyTypeChecker
    args.activation = _configure_activation_function(args.activation)
    args.detector = BlackBoxDetector.Type.get_from_name(args.detector)

    # Check the malware and binary files exist
    args.mal_file = Path(args.mal_file)
    args.ben_file = Path(args.ben_file)
    for (name, path) in (("malware", args.mal_file), ("benign", args.ben_file)):
        if path.exists(): continue
        print(f"Unknown %s file \"%s\"" % (name, str(path)))
        sys.exit(1)
    return args


def _configure_activation_function(act_func_name: str) -> nn.Module:
    r"""
    Parse the activation function from a string and return the corresponding activation function
    PyTorch module.  If the activation function cannot not be found, a \p ValueError is thrown.

    **Note**: Activation function check is case insensitive.

    :param act_func_name: Name of the activation function to
    :return: Activation function module associated with the passed name.
    """
    act_func_name = act_func_name.lower()  # Make case insensitive
    # Supported activation functions
    act_funcs = [("relu", nn.ReLU), ("elu", nn.ELU), ("leakyrelu", nn.LeakyReLU), ("tanh", nn.Tanh),
                 ("sigmoid", nn.Sigmoid)]
    for func_name, module in act_funcs:
        if act_func_name == func_name.lower():
            return module
    raise ValueError("Unknown activation function: \"%s\"" % act_func_name)


def load_dataset(file_path: Union[str, Path], y: int) -> MalwareDataset:
    r"""
    Extracts the input data from disk and packages them into format expected by \p MalGAN.  Supports
    loading files from numpy, torch, and pickle.  Other formats (based on the file extension) will
    result in a \p ValueError.

    :param file_path: Path to a NumPy data file containing tensors for the benign and malware
                      data.
    :param y: Y value for dataset
    :return: \p MalwareDataset objects for the malware and benign files respectively.
    """
    file_ext = Path(file_path).suffix
    if file_ext in {".npy", ".npz"}:
        data = np.load(file_path)
    elif file_ext in {".pt", ".pth"}:
        data = torch.load(str(file_path))
    elif file_ext == ".pk":
        with open(str(file_path), "rb") as f_in:
            data = pickle.load(f_in)
    else:
        raise ValueError("Unknown file extension.  Cannot determine how to import")
    return MalwareDataset(x=data, y=y)


def main(list_of_apis_to_use=None):
    if list_of_apis_to_use is None:
        list_of_apis_to_use = []

    args = parse_args()
    setup_logger(args.q)

    MalGAN.MALWARE_BATCH_SIZE = args.batch_size

    malgan = MalGAN(load_dataset(args.mal_file, MalGAN.Label.Malware.value),
                    load_dataset(args.ben_file, MalGAN.Label.Benign.value),
                    Z=args.Z,
                    h_gen=args.gen_hidden_sizes,
                    h_discrim=args.discrim_hidden_sizes,
                    g_hidden=args.activation,
                    detector_type=args.detector,
                    pth=args.pth)
    if not args.pth:
        malgan.fit(args.num_epoch, quiet_mode=args.q)
        results = malgan.measure_and_export_results()
        if args.print_results:
            print(results)

    malgan = malgan.cpu()

    binary_array, _ = malgan(next(iter(malgan._mal_data.test))[0][:1])
    array = binary_array.flatten().cpu().numpy()

    for func in list_of_apis_to_use:
        index = api_map_to_index[func]

        array[index] = 1

    genscript.generate(array)


if __name__ == "__main__":
    main_fct = main

    app = QApplication(sys.argv)
    w = AppWindow(main_fct)
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

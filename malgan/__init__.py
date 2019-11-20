import logging
import os
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader, Subset

from .detector import BlackBoxDetector
from .discriminator import Discriminator
from .generator import Generator

ListOrInt = Union[List[int], int]
PathOrStr = Union[str, Path]
TensorTuple = Tuple[Tensor, Tensor]

IS_CUDA_AVAILABLE = torch.cuda.is_available()
if IS_CUDA_AVAILABLE:
    device = torch.device('cuda:0')

    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class MalwareDataset(Dataset):
    def __init__(self, data: Union[np.ndarray, Tensor], classes):
        super().__init__()

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        self.x = data
        self.y = classes

    def __getitem__(self, index):
        return self.x[index], self.y

    def __len__(self):
        return self.x.shape[0]

    @property
    def num_features(self):
        return self.x.shape[1]


class _DataGroup:
    def __init__(self, train_dataset: MalwareDataset, validation_dataset: MalwareDataset, test_dataset: MalwareDataset):
        self.train = train_dataset
        self.valid = validation_dataset
        self.test = test_dataset
        self.is_loaders = False

    def build_loader(self, train_batch_size: int = 0):
        self.train = DataLoader(self.train, batch_size=train_batch_size, shuffle=True)
        if self.valid:
            self.valid = DataLoader(self.valid, batch_size=train_batch_size)
        self.test = DataLoader(self.test, batch_size=train_batch_size)
        self.is_loaders = True


class MalGAN(nn.Module):
    MALWARE_BATCH_SIZE = 32

    SAVED_MODEL_DIR = Path("saved_models")

    VALIDATION_SPLIT = 0.2

    tensorboard = None

    class Label(Enum):
        Malware = 1
        Benign = 0

    def __init__(self, malware_data: MalwareDataset, benign_data: MalwareDataset, dim_noise_vect: int,
                 hidden_layer_width_generator: ListOrInt, hidden_layer_width_discriminator: ListOrInt,
                 test_split: float = 0.2,
                 generator_hidden: nn.Module = nn.LeakyReLU,
                 detector_type: BlackBoxDetector.Type = BlackBoxDetector.Type.LogisticRegression,
                 path: Path = None):

        super().__init__()

        if malware_data.num_features != benign_data.num_features:
            raise ValueError("Mismatch in the number of features between malware and benign data")

        if dim_noise_vect <= 0:
            raise ValueError("Z must be a positive integers")

        if test_split <= 0. or test_split >= 1.:
            raise ValueError("test_split must be in the range (0,1)")

        self._M, self._Z = malware_data.num_features, dim_noise_vect

        if isinstance(hidden_layer_width_generator, int):
            hidden_layer_width_generator = [hidden_layer_width_generator]

        if isinstance(hidden_layer_width_discriminator, int):
            hidden_layer_width_discriminator = [hidden_layer_width_discriminator]

        self.d_discrim, self.d_gen = hidden_layer_width_discriminator, hidden_layer_width_generator

        for hidden_layer_widths in [self.d_discrim, self.d_gen]:
            for width in hidden_layer_widths:
                if width <= 0:
                    raise ValueError("All hidden layer widths must be positive integers.")

        if not isinstance(generator_hidden, nn.Module):
            generator_hidden = generator_hidden()
        self._g = generator_hidden

        self._is_cuda = IS_CUDA_AVAILABLE

        print("Constructing new MalGAN")
        print("Malware Dimension (M): %d", self.M)
        print("Latent Dimension (Z): %d", self.Z)
        print("Test Split Ratio: %.3f", test_split)
        print("Generator Hidden Layer Sizes: %s", hidden_layer_width_generator)
        print("Discriminator Hidden Layer Sizes: %s", hidden_layer_width_discriminator)
        print("Blackbox Detector Type: %s", detector_type.name)
        print("Activation Type: %s", self._g.__class__.__name__)

        self._bb = BlackBoxDetector(detector_type)
        self._gen = Generator(dim_feature_vect=self.M, dim_noise_vect=self.Z, hidden_size=hidden_layer_width_generator, activation_fct=self._g)
        self._discrim = Discriminator(width_of_malware=self.M, hidden_layer_size=hidden_layer_width_discriminator, activation_fct=self._g)

        def split_train_valid_test(dataset: Dataset, is_benign: bool):
            validation_size = 0 if is_benign else int(MalGAN.VALIDATION_SPLIT * len(dataset))
            test_size = int(test_split * len(dataset))

            lengths = [len(dataset) - validation_size - test_size, validation_size, test_size]
            return _DataGroup(*torch.utils.data.random_split(dataset, lengths))

        self._mal_data = split_train_valid_test(malware_data, is_benign=False)
        self._ben_data = split_train_valid_test(benign_data, is_benign=True)

        self._fit_blackbox(self._mal_data.train, self._ben_data.train)

        self._mal_data.build_loader(MalGAN.MALWARE_BATCH_SIZE)
        ben_bs_frac = len(benign_data) / len(malware_data)
        self._ben_data.build_loader(int(ben_bs_frac * MalGAN.MALWARE_BATCH_SIZE))

        if path:
            self.load(path)

        if self._is_cuda:
            self.cuda()

    @property
    def M(self) -> int:
        return self._M

    @property
    def Z(self) -> int:
        return self._Z

    def _fit_blackbox(self, malware_training: Subset, benign_training: Subset) -> None:
        def extract_x(dataset: Subset) -> Tensor:
            x = dataset.dataset.x[dataset.indices]
            return x.cpu() if self._is_cuda else x

        malware_data = extract_x(malware_training)
        benign_data = extract_x(benign_training)
        merged_data = torch.cat((malware_data, benign_data))

        merged_classes = torch.cat((torch.full((len(malware_training),), MalGAN.Label.Malware.value),
                              torch.full((len(benign_training),), MalGAN.Label.Benign.value)))
        print("Starting training of blackbox detector of type \"%s\"", self._bb.type.name)

        self._bb.fit(merged_data, merged_classes)
        print("COMPLETED training of blackbox detector of type \"%s\"", self._bb.type.name)

    def fit(self, number_of_epochs: int) -> None:
        if number_of_epochs <= 0:
            raise ValueError("At least a single training cycle is required.")

        MalGAN.tensorboard = tensorboardX.SummaryWriter()

        discriminator_optimizer = optim.Adam(self._discrim.parameters(), lr=1e-5)
        generator_optimizer = optim.Adam(self._gen.parameters(), lr=1e-4)

        best_epoch, best_loss = None, np.inf
        for epoch_count in range(1, number_of_epochs + 1):
            train_loss_generator, train_loss_discriminator = self._fit_epoch(generator_optimizer, discriminator_optimizer)
            for block, loss in [("Generator", train_loss_generator), ("Discriminator", train_loss_discriminator)]:
                MalGAN.tensorboard.add_scalar('Train_%s_Loss' % block, loss, epoch_count)

            validation_loss_generator = self._meas_loader_gen_loss(self._mal_data.valid)
            MalGAN.tensorboard.add_scalar('Validation_Generator_Loss', validation_loss_generator, epoch_count)
            fields = [train_loss_generator, validation_loss_generator, train_loss_discriminator, validation_loss_generator < best_loss]
            if fields[-1]:
                self._save(self._build_export_name(is_final=False))
                best_loss = validation_loss_generator
        MalGAN.tensorboard.close()

        self.load(self._build_export_name(is_final=False))
        self._save(self._build_export_name(is_final=True))
        self._delete_old_backup(is_final=False)

    def _build_export_name(self, is_final: bool = True) -> str:
        name = ["malgan", "z=%d" % self.Z,
                "d-gen=%s" % str(self.d_gen).replace(" ", "_"),
                "d-disc=%s" % str(self.d_discrim).replace(" ", "_"),
                "bs=%d" % MalGAN.MALWARE_BATCH_SIZE,
                "bb=%s" % self._bb.type.name, "g=%s" % self._g.__class__.__name__,
                "final" if is_final else "tmp"]

        return MalGAN.SAVED_MODEL_DIR / "".join(["_".join(name).lower(), ".pth"])

    def _delete_old_backup(self, is_final: bool = True) -> None:
        backup_name = self._build_export_name(is_final)
        try:
            os.remove(backup_name)
        except OSError:
            print("Error trying to delete model: %s", backup_name)

    def _fit_epoch(self, generator_optimizer: Optimizer, discriminator_optimizer: Optimizer) -> TensorTuple:
        total_loss_generator = total_loss_discriminator = 0
        num_batch = min(len(self._mal_data.train), len(self._ben_data.train))

        for (malware_data, _), (benign_data, _) in zip(self._mal_data.train, self._ben_data.train):
            if self._is_cuda:
                malware_data, benign_data = malware_data.cuda(), benign_data.cuda()
            m_prime, g_theta = self._gen.forward(malware_data)
            generator_loss = self._calc_gen_loss(g_theta)
            generator_optimizer.zero_grad()
            generator_loss.backward()

            generator_optimizer.step()
            total_loss_generator += generator_loss

            for x in [m_prime, benign_data]:
                discriminator_loss = self._calc_discrim_loss(x)
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()

                discriminator_optimizer.step()
                total_loss_discriminator += discriminator_loss
        # noinspection PyUnresolvedReferences
        return (total_loss_generator / num_batch).item(), (total_loss_discriminator / num_batch).item()

    def _meas_loader_gen_loss(self, loader: DataLoader) -> float:
        r""" Calculate the generator loss on malware dataset """
        loss = 0
        for m, _ in loader:
            if self._is_cuda: m = m.cuda()
            _, g_theta = self._gen.forward(m)
            loss += self._calc_gen_loss(g_theta)
        # noinspection PyUnresolvedReferences
        return (loss / len(loader)).item()

    def _calc_gen_loss(self, g_theta: Tensor) -> Tensor:
        d_theta = self._discrim.forward(g_theta)

        return d_theta.log().mean()

    def _calc_discrim_loss(self, examples_to_calculate_loss: Tensor) -> Tensor:
        d_theta = self._discrim.forward(examples_to_calculate_loss)

        y_hat = self._bb.predict(examples_to_calculate_loss)

        d = torch.where(y_hat == MalGAN.Label.Malware.value, d_theta, 1 - d_theta)
        return -d.log().mean()

    def _save(self, file_path: PathOrStr) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(file_path))

    def forward(self, malware_binary_tensor: Tensor) -> TensorTuple:
        return self._gen.forward(malware_binary_tensor)

    def load(self, file_path: PathOrStr) -> None:
        if isinstance(file_path, Path):
            file_path = str(file_path)

        self.load_state_dict(torch.load(file_path))
        self.eval()

        for module in self._gen.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

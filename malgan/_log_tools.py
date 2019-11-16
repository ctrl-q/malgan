import copy
import logging
import sys
from _decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Any

import torch
from torch import Tensor

ListOrInt = Union[int, List[int]]

LOG_DIRECTORY = Path(".")
IS_CUDA_AVAILABLE = torch.cuda.is_available()


def setup_logger(is_in_quiet_mode: bool, log_level: int = logging.DEBUG,
                 job_id: Optional[ListOrInt] = None) -> None:
    date_format = '%m/%d/%Y %I:%M:%S %p'
    format_string = '%(asctime)s -- %(levelname)s -- %(message)s'

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    fields = ["logs"]
    if job_id is not None:
        if isinstance(job_id, int):
            job_id = [job_id]
        fields += ["_j=", "-".join("%05d" % id for id in job_id)]

    fields += ["_", str(datetime.now()).replace(" ", "-"), ".log"]

    filename = LOG_DIRECTORY / "".join(fields)
    logging.basicConfig(filename=filename, level=log_level, format=format_string, datefmt=date_format)

    if not is_in_quiet_mode:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    logging.info("******************* New Run Beginning *****************")
    logging.debug("CUDA: %s", "ENABLED" if IS_CUDA_AVAILABLE else "Disabled")
    logging.info(" ".join(sys.argv))


class TrainingLogger:
    r""" Helper class used for standardizing logging """
    FIELD_SEPARATOR = " "
    DEFAULT_WIDTH = 12
    EPOCH_WIDTH = 5

    DEFAULT_FIELD = None

    LOG = logging.info

    def __init__(self, field_names: List[str], field_widths: Optional[List[int]] = None):
        if field_widths is None:
            field_widths = len(field_names) * [TrainingLogger.DEFAULT_WIDTH]
        if len(field_widths) != len(field_names):
            raise ValueError("Mismatch in the length of field names and widths.")

        self._log = TrainingLogger.LOG
        self._field_widths = field_widths

        combined_names = ["Epoch"] + field_names
        combined_widths = [TrainingLogger.EPOCH_WIDTH] + field_widths
        format_string = TrainingLogger.FIELD_SEPARATOR.join(["{:^%d}" % width for width in combined_widths])
        self._log(format_string.format(*combined_names))

        separator_line = TrainingLogger.FIELD_SEPARATOR.join(["{:-^%d}" % width for width in combined_widths])
        logging.info(separator_line.format(*(len(combined_widths) * [""])))

    @property
    def num_fields(self) -> int:
        return len(self._field_widths)

    def log(self, epoch: int, values: List[Any]) -> None:
        values = self._clean_values_list(values)
        format_string = self._build_values_format_str(values)
        self._log(format_string.format(epoch, *values))

    def _build_values_format_str(self, values: List[Any]) -> str:
        def _get_fmt_str(_w: int, fmt: str) -> str:
            return "{:^%d%s}" % (_w, fmt)

        format_string = [_get_fmt_str(self.EPOCH_WIDTH, "d")]
        for width, value in zip(self._field_widths, values):
            if isinstance(value, str):
                format_chars = "s"
            elif isinstance(value, Decimal):
                format_chars = ".3E"
            elif isinstance(value, int):
                format_chars = "d"
            elif isinstance(value, float):
                format_chars = ".4f"
            else:
                raise ValueError("Unknown value type")

            format_string.append(_get_fmt_str(width, format_chars))
        return TrainingLogger.FIELD_SEPARATOR.join(format_string)

    def _clean_values_list(self, values: List[Any]) -> List[Any]:
        values = copy.deepcopy(values)

        while len(values) < self.num_fields:
            values.append(TrainingLogger.DEFAULT_FIELD)

        new_values = []
        for value in values:
            if isinstance(value, bool):
                value = "+" if value else ""
            elif value is None:
                value = "N/A"
            elif isinstance(value, Tensor):
                value = value.item()

            if isinstance(value, float) and (value <= 1E-3 or value >= 1E4):
                value = Decimal(value)
            new_values.append(value)
        return new_values

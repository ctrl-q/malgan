import datetime
from pathlib import Path
from typing import Union

import numpy as np

import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

TensorOrFloat = Union[torch.Tensor, float]
TorchOrNumpy = Union[torch.Tensor, np.ndarray]


# noinspection PyProtectedMember,PyUnresolvedReferences
def _export_results(model: 'MalGAN', avg_validation_loss: TensorOrFloat, avg_test_loss: TensorOrFloat,
                    avg_num_bits_changed: TensorOrFloat, actual_labels: np.ndarray,
                    predicted_value_original_malware: TorchOrNumpy, prob_of_malware: TorchOrNumpy, predicted_labels: np.ndarray) -> str:

    if isinstance(prob_of_malware, torch.Tensor):
        prob_of_malware = prob_of_malware.numpy()
    if isinstance(predicted_value_original_malware, torch.Tensor):
        predicted_value_original_malware = predicted_value_original_malware.numpy()

    results_file = Path("results.csv")
    file_already_exist = results_file.exists()
    with open(results_file, "a+") as out_file:
        header = ",".join(["time_completed,M,Z,batch_size,test_set_size,detector_type,activation",
                           "gen_hidden_dim,discim_hidden_dim",
                           "avg_validation_loss,avg_test_loss,avg_num_bits_changed",
                           "auc,orig_mal_detect_rate,mod_mal_detect_rate,ben_mal_detect_rate"])
        if not file_already_exist:
            out_file.write(header)

        results = ["\n%s" % datetime.datetime.now(),
                   "%d,%d,%d" % (model.M, model.Z, model.__class__.MALWARE_BATCH_SIZE),
                   "%d,%s,%s" % (len(actual_labels), model._bb.type.name, model._g.__class__.__name__),
                   "\"%s\",\"%s\"" % (str(model.d_gen), str(model.d_discrim)),
                   "%.15f,%.15f,%.3f" % (avg_validation_loss, avg_test_loss, avg_num_bits_changed)]

        auc = roc_auc_score(actual_labels, prob_of_malware)
        results.append("%.8f" % auc)

        results.append("%.8f" % predicted_value_original_malware.mean())

        true_positive, false_positive, false_negative, true_positive = confusion_matrix(actual_labels, predicted_labels).ravel()
        true_positive_ratio, false_positive_ratio = true_positive / (true_positive + false_negative), false_positive / (true_positive + false_positive)
        for rate in [true_positive_ratio, false_positive_ratio]:
            results.append("%.8f" % rate)
        results = ",".join(results)
        out_file.write(results)

        return "".join([header, results])

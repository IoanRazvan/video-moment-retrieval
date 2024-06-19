import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
import os
import jsonlines
from typing import Any
import numpy.typing as npt
from video_moment_retrieval.utils.utils import edges_to_center

QVDataPoint = tuple[npt.NDArray[np.float32],
                    npt.NDArray[np.float32], None | dict[str, Any], dict[str, Any]]


class QVDataset(Dataset):
    def __init__(self, text_features_path: str, video_features_path: str, data_file_path: str):
        super().__init__()
        self.text_features_path = text_features_path
        self.video_features_path = video_features_path
        self.data = []
        with jsonlines.open(data_file_path) as reader:
            for el in reader:
                self.data.append(el)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> QVDataPoint:
        item_info = self.data[index]
        qid = item_info["qid"]
        vid = item_info["vid"]
        meta = {
            "vid": vid,
            "qid": qid,
            "query": item_info["query"],
            "duration": item_info["duration"]
        }
        with np.load(os.path.join(self.text_features_path, str(qid) + ".npz")) as np_archive:
            query_features = np_archive["last_hidden_state"]
        video_features = np.load(os.path.join(
            self.video_features_path, vid + ".npy"))
        eps = 1e-5
        video_features = video_features / \
            (np.linalg.norm(video_features, axis=-1, keepdims=True) + eps)
        query_features = query_features / \
            (np.linalg.norm(query_features, axis=-1, keepdims=True) + eps)

        label = None

        if "relevant_clip_ids" in item_info:
            label = {
                "boxes": np.array([edges_to_center(window) for window in item_info["relevant_windows"]], dtype=np.float32) / item_info["duration"],
                "class_labels": np.zeros((len(item_info["relevant_windows"]),), dtype=np.int64),
                "relevant_clip_ids": np.array(item_info["relevant_clip_ids"], dtype=np.int32),
                "saliency_scores": np.array(item_info["saliency_scores"]).mean(axis=1),
                "relevant_windows": np.array(item_info["relevant_windows"], dtype=np.int32),
                "duration": np.array(item_info["duration"])
            }
        return (query_features, video_features, label, meta)


def pad_sequence(sequence: list[npt.NDArray]) -> torch.Tensor:
    batch_size, seq_len, hidden = len(sequence), max(
        feat.shape[0] for feat in sequence), sequence[0].shape[-1]
    result = torch.zeros((batch_size, seq_len, hidden))
    attn = torch.zeros((batch_size, seq_len))
    for idx, sample in enumerate(sequence):
        result[idx, :sample.shape[0], :] = torch.tensor(sample)
        attn[idx, :sample.shape[0]] = 1

    return result, attn


def pad_collate(samples: list[QVDataPoint]):
    text_features, text_attn_mask = pad_sequence([el[0] for el in samples])
    video_features, video_attn_mask = pad_sequence([el[1] for el in samples])

    labels = None
    if samples[0][2] is not None:
        labels = []
        for (_, _, sample, _) in samples:
            labels.append(
                { k: torch.from_numpy(v) for k, v in sample.items() }
            )

    return {
        "text_features": text_features,
        "text_attn_mask": text_attn_mask,
        "video_features": video_features,
        "video_attn_mask": video_attn_mask,
        "labels": labels,
        "meta": [sample[3] for sample in samples]
    }

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
import os
import jsonlines
from typing import Any
import numpy.typing as npt
from video_moment_retrieval.utils.utils import edges_to_center
import random

QVDataPoint = tuple[npt.NDArray[np.float32],
                    npt.NDArray[np.float32], None | dict[str, Any], dict[str, Any]]


class QVDataset(Dataset):
    def __init__(self, text_features_path: str, video_features_path: str, data_file_path: str):
        super().__init__()
        self.text_features_path = text_features_path
        self.video_features_path = video_features_path
        self.data = []
        self.max_windows = 5
        self.max_q_l = 32
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
        ctx_len = video_features.shape[0]
        tef_st = np.arange(0, ctx_len) / ctx_len
        tef_ed = tef_st + 1.0 / ctx_len
        tef = np.stack([tef_st, tef_ed], axis=1)
        video_features = video_features / \
            (np.linalg.norm(video_features, axis=-1, keepdims=True) + eps)
        video_features = np.concatenate([video_features, tef], axis=-1)
        query_features = query_features[:self.max_q_l]
        query_features = query_features / \
            (np.linalg.norm(query_features, axis=-1, keepdims=True) + eps)

        label = None

        if "relevant_clip_ids" in item_info:
            windows = item_info["relevant_windows"]
            random.shuffle(windows)
            windows = windows[:self.max_windows]
            label = {
                "boxes": np.array([edges_to_center(window) for window in windows], dtype=np.float32) / item_info["duration"],
                "class_labels": np.zeros((len(windows),), dtype=np.int64),
                **(self._get_saliency_labels(item_info)),
                "relevant_windows": np.array(windows, dtype=np.int32),
                "duration": np.array(item_info["duration"])
            }
        return (query_features, video_features, label, meta)

    def _get_saliency_labels(self, vid_info: dict[str, Any]) -> dict[str, npt.NDArray]:
        ctx_l = 75
        scores = np.array(vid_info["saliency_scores"]).sum(axis=0)
        high_idx, low_idx = np.argmax(scores), np.argmin(scores)
        positive_ids, negative_ids = [high_idx], [low_idx]
        outside_clips = set(range(ctx_l)) - set(vid_info["relevant_clip_ids"])
        if outside_clips:
            positive_ids += random.sample(vid_info["relevant_clip_ids"], k=1)
            negative_ids += random.sample(list(outside_clips), k=1)
        else:
            positive_ids += positive_ids
            negative_ids += negative_ids
            
        return {
            "positive_ids": np.array(positive_ids, dtype=np.int32),
            "negative_ids": np.array(negative_ids, dtype=np.int32)
        }


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

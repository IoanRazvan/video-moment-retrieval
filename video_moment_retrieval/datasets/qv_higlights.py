import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
import os
import jsonlines
from typing import Any

QVDataPointTarget = dict[str, Any]
QVDataPoint = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], None | QVDataPointTarget]

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
        query_features = np.load(os.path.join(self.text_features_path, str(qid) + ".npy"))
        video_features = np.load(os.path.join(self.video_features_path, vid + ".npy"))
        label = None
        if "relevant_clip_ids" in item_info:
            label = {
                "relevant_clip_ids": item_info["relevant_clip_ids"], 
                "saliency_scores": item_info["saliency_scores"], 
                "relevant_windows": item_info["relevant_windows"],
                "duration": item_info["duration"]
            }
        return (query_features, video_features, label)

# define Data collator function
def pad_collate(samples: list[QVDataPoint]):
    # Need to: pad video and text features -> create attention video/text attention
    text_lens = [el[0].shape[0] for el in samples]
    video_lens = [el[1].shape[0] for el in samples]
    
    batch_size = len(samples)
    text_len = max(text_lens)
    text_hidden = samples[0][0].shape[-1]
    video_len = max(video_lens)
    video_hidden = samples[0][1].shape[-1]
    
    text_features = torch.zeros((batch_size, text_len, text_hidden))
    text_attn_mask = torch.ones((batch_size, text_len))
    video_features = torch.zeros((batch_size, video_len, video_hidden))
    video_attn_mask = torch.ones((batch_size, video_len))
    
    for idx, sample in enumerate(samples):
        sample_text_len = sample[0].shape[0]
        sample_video_len = sample[1].shape[0]
        text_features[idx, :sample_text_len, :] = torch.tensor(sample[0])
        video_features[idx, :sample_video_len, :] = torch.tensor(sample[1])
        text_attn_mask[idx, sample_text_len:] = 0
        video_attn_mask[idx, sample_video_len:] = 0
    
    labels = None
    # We have labels
    if samples[0][2] is not None:
        # build classes array
        labels = []
        for (_, _, sample) in samples:
            duration = sample["duration"]
            boxes = sample["relevant_windows"]
            class_labels = torch.zeros((len(boxes), ), dtype=torch.int64) 
            labels.append({
                "boxes": torch.tensor([[box[0] / duration, box[1] / duration] for box in boxes]),
                "class_labels": class_labels
            })
    
    return  {
        "text_features": text_features,
        "text_attn_mask": text_attn_mask,
        "video_features": video_features,
        "video_attn_mask": video_attn_mask,
        "labels": labels
    }
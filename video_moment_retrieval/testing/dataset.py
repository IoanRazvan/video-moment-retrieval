from torch.utils.data import Dataset, DataLoader
import os
import json
import decord
decord.bridge.set_bridge("torch")
from decord import VideoReader
import numpy as np
import time

class VideoDataset(Dataset):
    def __init__(self, annotations_file: str, video_root: str, n_frames: int = 60):
        with open(annotations_file, "r") as f:
            self.data = [json.loads(l) for l in f.readlines() if l]
        self.video_root = video_root
        self.n_frames = n_frames
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_item = self.data[index]
        video_path = os.path.join(self.video_root, data_item["vid"] + ".mp4")
        vr = VideoReader(video_path, width=224, height=224, num_threads=4)
        
        total_frames = len(vr)
        
        intervals = np.linspace(0, total_frames, self.n_frames, endpoint=False, dtype=int)
        
        frame_indices = [int(l + r) // 2 for l, r in zip(intervals[0:], intervals[1:])]
    
        return vr.get_batch(frame_indices)
    
if __name__ == "__main__":
    ds = VideoDataset("qvhighlights_features/highlight_train_release.jsonl", "../videos")
    dataloader = DataLoader(ds, num_workers=10, batch_size=32, persistent_workers=True, prefetch_factor=1)
    
    start = time.time()
    for el in dataloader:
        end = time.time()
        print("Reading batch took %fs" % (end - start))
        start = end
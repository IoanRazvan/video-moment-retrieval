import timm
import torch
from transformers import BertTokenizer, BertModel
import json
import numpy as np
import os
import tqdm
from decord import VideoReader
import numpy.typing as npt

from concurrent.futures import ThreadPoolExecutor, as_completed


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
resnet50 = timm.create_model("resnet50", pretrained=True).eval().to("cuda")
data_transform = timm.data.create_transform(**timm.data.resolve_data_config(resnet50.pretrained_cfg), is_training=False)

input_root = "../videos/"
output_root = "bert_features"
output_video_root = "resnet_features"
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_video_root, exist_ok=True)

@torch.no_grad()
def process_query(el: dict):
    input_ids = tokenizer(el["query"], return_tensors="pt")
    input_ids = {k: v.to("cuda") for k, v in input_ids.items()}
    outputs = model(**input_ids)
    embeddings = outputs.last_hidden_state[0, 1:-1, :]
    np.save(f"{os.path.join(output_root, str(el['qid']))}.npy", embeddings.cpu().numpy())

def read_video(el: dict, num_frames: int):
    video_path = os.path.join(input_root, el["vid"] + ".mp4")
    feat_path = os.path.join(output_video_root, el["vid"] + ".npz")
    if os.path.exists(feat_path):
        try:
            np.load(feat_path)["features"]
            return None
        except:
            pass
    vr = VideoReader(video_path, num_threads=4)
    total_frames = len(vr)
    intervals = np.linspace(0, total_frames, num_frames + 1, endpoint=False, dtype=int)
    frame_indices = [int(l + r) // 2 for l, r in zip(intervals[0:], intervals[1:])]

    frames = vr.get_batch(frame_indices).asnumpy()
    return frames, np.array(frame_indices), el

@torch.no_grad()
def process_frames(el: dict, frames: npt.NDA, indices):
    try:
        video_path = os.path.join(output_video_root, el["vid"] + ".npz")
        
        frames = torch.tensor(frames).permute((0, 3, 1, 2)) / 255.0
        frames = frames.to("cuda")
        frames = data_transform(frames)
        output = resnet50.forward_features(frames)
        np.savez_compressed(video_path, features=output.cpu().numpy(), indices=indices)

    except Exception as e:
        print(e)
    
    
def main():
    with open("qvhighlights_features/highlight_train_release.jsonl", "r") as f:
        data = [json.loads(l) for l in f.readlines() if l]
    with open("qvhighlights_features/highlight_val_release.jsonl", "r") as f:
        data.extend([json.loads(l) for l in f.readlines() if l])
    print(len(data))
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(read_video, el, 100) for el in data]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if not res:
                continue
            frames, indices, el = res
            process_frames(el, frames, indices)
            
    # for el in tqdm.tqdm(data[1:]):
    #     process_query(el)
        
if __name__ == "__main__":
    main()
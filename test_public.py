import sys
sys.path.append("..\\moment_detr")
from run_on_video.model_utils import build_inference_model
import torch

from video_moment_retrieval.datasets.qv_highlights import QVDataset, pad_collate
from torch.utils.data import DataLoader
from video_moment_retrieval.detr_matcher.matcher import VideoDetrHungarianMatcher, VideoDetrLoss
import tqdm

train_dataset = QVDataset("qvhighlights_features\\text_features", "qvhighlights_features\\video_features", "qvhighlights_features\\highlight_train_release.jsonl")
train_loader = DataLoader(train_dataset, 32, False, collate_fn=pad_collate)

model = build_inference_model("..\\moment_detr\\run_on_video\\moment_detr_ckpt\\model_best.ckpt")
model.eval()

weight_dict = {
    "loss_ce": 4,
    "loss_bbox": 10,
    "loss_giou": 1
}

losses = []

matcher = VideoDetrHungarianMatcher(4, 10, 1)
criterion = VideoDetrLoss(matcher, 1, 0.1, ["labels", "boxes"])

with torch.no_grad():
    for batch in tqdm.tqdm(train_loader):
        labels = batch["labels"]
        del batch["labels"]
        output = model(**batch)
        criterion_input = {
            "logits": output["pred_logits"],
            "pred_boxes": output["pred_spans"]
        }
        loss_dict = criterion(criterion_input, labels)
        
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        losses.append(loss.item())

print(sum(losses) / len(losses))
import numpy as np

from transformers import PreTrainedModel, VivitModel, VivitImageProcessor, BertTokenizerFast, BertModel, DetrConfig
from transformers.models.detr.modeling_detr import DetrDecoder
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Mapping, Optional
from torch.utils.data import default_collate
from itertools import chain
import torch
from video_moment_retrieval.qd_detr.model import CrossAttentionEncoder
from video_moment_retrieval.moment_detr.model import VideoDetrConfig, VideoDetrModelOutput, VideoDetrHungarianMatcher, VideoDetrLoss
from video_moment_retrieval.utils.utils import edges_to_center, extract_frames_ffmpeg
import torch.nn as nn
import numpy.typing as npt
import time

class VideoDataset(Dataset):
    def __init__(self, annotations_file: str, videos_root: str, cache_root: str, frame_sampling: str = "uniform", n_frames: int = 60, height: int = 224, width: int = 224, max_query_len: int = 32):
        super().__init__()
        self.cache_root = cache_root
        with open(annotations_file) as f:
            self.data = [json.loads(line) for line in f.readlines() if line]
        self.videos_root = videos_root
        self.frame_sampling = frame_sampling
        self.n_frames = n_frames
        self.width = width
        self.height = height
        self.max_query_len = max_query_len
        self.vivit_processor = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(
            "google-bert/bert-base-uncased")
    
    def __len__(self):
        return len(self.data)
    
    # TODO: Assume a generic format rather than the QVHighlights one
    def __getitem__(self, index: int):
        data_item = self.data[index]
        video_path = os.path.join(self.videos_root, data_item["vid"]) + ".mp4"
        if not os.path.isfile(video_path):
            raise Exception("%s does not exist" % video_path)
        cached_file = os.path.join(self.cache_root, data_item["vid"] + ".npy")
        if os.path.exists(cached_file):
            frames = [frame for frame in np.load(cached_file)]
        else:
            frames = extract_frames_ffmpeg(
                video_path, self.frame_sampling, self.n_frames, self.width, self.height)
        processed = self.vivit_processor(frames, return_tensors="pt", do_resize=True, size={"width": 224, "height": 224})
        tokenized_query = self.bert_tokenizer(
            data_item["query"], max_length=self.max_query_len, padding="max_length", truncation=True, return_tensors="pt")
        duration = data_item["duration"]
        relevant_windows = data_item["relevant_windows"]
        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            **{k: v if not isinstance(v, torch.Tensor) else v.squeeze(0) for k, v in tokenized_query.items()},
            "labels": {
                "boxes": np.array([edges_to_center(window) for window in relevant_windows], dtype=np.float32) / duration,
                "relevant_windows": np.array(relevant_windows, dtype=np.int32),
                "class_labels": np.zeros((len(relevant_windows),), dtype=np.int64),
                "duration": np.array(duration),
            }
        }

    @staticmethod
    def collate(batch: List[Mapping]):
        # echo for torch default_collate with the exception of lists where instead of converting to NumPy
        # and requiring each list dimension match we only create a list of lists
        first_elem = batch[0]
        collated = {
            k: default_collate([el[k] for el in batch]) for k in first_elem.keys() if k != "labels"
        }
        labels = [
            {k: torch.tensor(v) for k, v in el["labels"].items()} for el in batch
        ]
        collated["labels"] = labels
        return collated


class ViVitRetrieval(PreTrainedModel):
    def __init__(self, config: VideoDetrConfig):
        super().__init__(config)
        self.config = config
        
        ### Init frozen video and query encoders 
        video_encoder = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", add_pooling_layer=False).eval()
        query_encoder = BertModel.from_pretrained(
            "google-bert/bert-base-uncased").eval()
        for param in chain(video_encoder.parameters(), query_encoder.parameters()):
            param.requires_grad_(False)
        self.video_encoder = video_encoder
        self.video_encoder.half()
        self.video_encoder.train = lambda _: self.video_encoder  # return the same model
        self.query_encoder = query_encoder
        self.query_encoder.train = lambda _: self.query_encoder
        
        ### Init trainable layers
        self.vid_projection = nn.Linear(video_encoder.config.hidden_size, config.hidden_size)
        self.txt_projection = nn.Linear(query_encoder.config.hidden_size, config.hidden_size)
        
        self.ca_encoder = CrossAttentionEncoder(
            2, config.d_model, config.encoder_attention_heads, config.encoder_ffn_dim, dropout=config.dropout
        )

        self.detr_decoder = DetrDecoder(config)
        self.moment_queries = nn.Embedding(config.num_queries, config.d_model)
        
        self.moment_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 2),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.post_init()
        
        
    def forward(self,
                pixel_values: torch.FloatTensor,
                input_ids: torch.LongTensor,
                token_type_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                labels: Optional[list[dict[str, torch.Tensor]]] = None,
                return_dict: Optional[bool] = None,
                **kwargs
            ):
        
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", dtype=torch.float16):
            tubelets = self.vid_projection(self.video_encoder(pixel_values).last_hidden_state[:, 1:, ...])
        tokens = self.vid_projection(self.query_encoder(input_ids, attention_mask, token_type_ids).last_hidden_state)
        
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", dtype=torch.float16):
            # Do cross attn between tublets and tokens, where q = tublets and k, v = query tokens
            qd_tubelets = self.ca_encoder(
                tubelets,
                tokens, 
                ~attention_mask.to(torch.bool)
            )
        
        object_queries = self.moment_queries.weight.unsqueeze(
            0).repeat(tubelets.shape[0], 1, 1)
        decoder_inputs = torch.zeros_like(object_queries)
        input_positions = torch.zeros_like(qd_tubelets)
        
        decoder_output = self.detr_decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=None,
            encoder_hidden_states=qd_tubelets,
            encoder_attention_mask=None,
            object_queries=input_positions,
            query_position_embeddings=object_queries
        )
        
        pred_moments = self.moment_predictor(
            decoder_output.last_hidden_state
        )
        logits = self.classifier(decoder_output.last_hidden_state)
        
        loss = None
        if labels is not None:
            outputs_loss = {
                "logits": logits,
                "pred_boxes": pred_moments
            }
            
            matcher = VideoDetrHungarianMatcher(
                self.config.class_cost, self.config.bbox_cost, self.config.giou_cost)
            criterion = VideoDetrLoss(
                matcher, self.config.num_labels, self.config.eos_coefficient, ["labels", "boxes"])
            criterion.to(self.device)
            loss_dict = criterion(outputs_loss, labels)
            weight_dict = {
                "loss_ce": self.config.ce_loss_coefficient,
                "loss_bbox": self.config.bbox_loss_coefficient,
                "loss_giou": self.config.giou_loss_coefficient,
                "loss_saliency": self.config.saliency_loss_coefficient
            }
            loss = sum(loss_dict[k] * weight_dict[k]
                       for k in loss_dict.keys() if k in weight_dict)
            

        output = (logits, pred_moments)
        if not return_dict:
            return (loss,) + output if loss else output

        return VideoDetrModelOutput(
            loss=loss,
            predicted_moments=pred_moments[-1],
            logits=logits[-1]
        )



if __name__ == "__main__":
    dataset = VideoDataset("qvhighlights_features/highlight_train_release.jsonl",
                           "../videos", "../frames", n_frames=32)
    data_loader = DataLoader(dataset, batch_size=2,
                             shuffle=True, collate_fn=VideoDataset.collate)
    start = time.time()
    batch = next(iter(data_loader))
    end = time.time()
    print("Batching took: %fs" % (end - start))
    batch = {k: v if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    config = VideoDetrConfig(
        d_model=256,
        encoder_layers=2,   
        encoder_ffn_dim=1024,
        decoder_layers=2,
        decoder_ffn_dim=1024,
        num_queries=10,
        dropout=0.1,
        activation_dropout=0.1,
        bbox_cost=10,
        giou_cost=1,
        class_cost=4,
        giou_loss_coefficient=1,
        bbox_loss_coefficient=10,
        num_labels=1
    )
    model = ViVitRetrieval(config)
    output = model(**batch)
    print(output)
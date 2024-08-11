from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast, T5ForConditionalGeneration
from transformers.utils import ModelOutput
import torch.nn as nn
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from video_moment_retrieval.moment_detr.model import positional_encodings
from typing import cast
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from torch.nn import Transformer
from video_moment_retrieval.utils.utils import count_parameters

class DDDataset(Dataset):
    def __init__(self, annotations_file: str, text_features_root: str, video_features_root: str, target_tokenizer: PreTrainedTokenizerFast, max_query_len: int = 32, max_relevant_windows: int = 5):
        super().__init__()
        with open(annotations_file) as f:
            self.data = [json.loads(line) for line in f.readlines() if line]
        self.max_query_len = max_query_len
        self.text_features_root = text_features_root
        self.video_features_root = video_features_root
        self.max_relevant_windows = max_relevant_windows
        self.target_tokenizer = target_tokenizer
        
    def __len__(self):
        return len(self.data)
    
    # TODO: Assume a generic format rather than the QVHighlights one
    def __getitem__(self, index: int):
        try:
            data_item = self.data[index]
            video_features_path = os.path.join(self.video_features_root, data_item["vid"]) + ".npz"
            text_features_path = os.path.join(self.text_features_root, str(data_item["qid"])) + ".npy"
            video_features = np.load(video_features_path)["features"]
            text_features = np.load(text_features_path)[:self.max_query_len]
        except Exception as e:
            print(data_item["qid"])
            raise e
        
        # TODO: normalize features?        
        # TODO: Somehow use timestamps as positional encodings
        relevant_windows = data_item["relevant_windows"][:self.max_relevant_windows]
        output = self.target_tokenizer(str(relevant_windows), return_tensors="pt")
        output = {k: v.squeeze(0) for k, v in output.items()}
        
        labels = output["input_ids"][1:]
        decoder_input_ids = output["input_ids"][:-1]
        decoder_attention_mask = output["attention_mask"][:-1]
        # decoder_attention_mask.masked_fill_(decoder_input_ids == self.target_tokenizer.eos_token_id, 0)
        # decoder_input_ids.masked_fill_(decoder_input_ids == self.target_tokenizer.eos_token_id, self.target_tokenizer.pad_token_id)
        
        return {
            "video_features": torch.tensor(video_features),
            "video_attention_mask": torch.ones(video_features.shape[0], dtype=torch.int32),
            "text_features": torch.tensor(text_features),
            "text_attention_mask": torch.ones(text_features.shape[0], dtype=torch.int32),
            # "timestamps": torch.tensor(timestamps),
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }
    @staticmethod
    def pad_sequence(sequences: list, pad_value: int, dtype: torch.dtype) -> torch.Tensor:
        max_len = max(len(sequence) for sequence in sequences)
        padded_sequences = torch.zeros((len(sequences), max_len, *sequences[0].shape[1:]), dtype=dtype) + pad_value
        for i, sequence in enumerate(sequences):
            padded_sequences[i, :len(sequence)] = sequence
        return padded_sequences
        
    def collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "video_features": DDDataset.pad_sequence([item["video_features"] for item in batch], 0, torch.float32),
            "video_attention_mask": DDDataset.pad_sequence([item["video_attention_mask"] for item in batch], 0, torch.int32),
            "text_features": DDDataset.pad_sequence([item["text_features"] for item in batch], 0, torch.float32),
            "text_attention_mask": DDDataset.pad_sequence([item["text_attention_mask"] for item in batch], 0, torch.int32),
            "decoder_input_ids": DDDataset.pad_sequence([item["decoder_input_ids"] for item in batch], self.target_tokenizer.pad_token_id, torch.int32),
            "decoder_attention_mask": DDDataset.pad_sequence([item["decoder_attention_mask"] for item in batch], 0, torch.int32),
            "labels": DDDataset.pad_sequence([item["labels"] for item in batch], self.target_tokenizer.pad_token_id, torch.int64),
        }

class DoubleDecoderConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 97,
        pad_token_id: int = 0,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        layer_norm_eps: float = 0.00001,
        num_layers: int = 2,
        video_features_shape: tuple[int, int] = (2048, 7),
        text_features_dim: int = 768,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.layer_norm_eps = layer_norm_eps
        self.video_features_shape = video_features_shape
        self.text_features_dim = text_features_dim
        self.pad_token_id_ = pad_token_id

        super().__init__(**kwargs)


@dataclass
class DoubleDecoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None

class DoubleDecoder(PreTrainedModel):
    def __init__(self, config: DoubleDecoderConfig):
        super().__init__(config)
        self.config = config
       
        self.out_vocab = nn.Embedding(self.config.vocab_size, self.config.d_model, self.config.pad_token_id_)
        self.text_projection = nn.Sequential(
            nn.Dropout1d(0.5),
            nn.Linear(self.config.text_features_dim, self.config.d_model)
        )
        self.frame_projection = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(
                self.config.video_features_shape[0],
                self.config.d_model,
                self.config.video_features_shape[1]
            )
        )

        self.feature_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            batch_first=True
        ), config.num_layers)
        
        self.moment_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            batch_first=True
        ), config.num_layers)

        self.seq_modeling_head = nn.Linear(self.config.d_model, self.config.vocab_size)
        
        self.post_init()

    def forward(
        self,
        video_features: torch.Tensor,
        video_attention_mask: torch.Tensor,
        text_features: torch.Tensor,
        text_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        # TODO: TODO: Add positional encodings where necessary
        bs, sq_l, channels, ks, _ = video_features.shape
        video_features = video_features.reshape(bs * sq_l, channels, ks, ks)
        video_features = self.frame_projection(video_features).flatten(1)
        video_features = video_features.reshape(bs, sq_l, -1)
        video_pos = positional_encodings(cast(torch.FloatTensor, video_features)).to(video_features.device)
        video_features = video_features + video_pos
        
        text_features = self.text_projection(text_features)
        
        target_tokens = self.out_vocab(decoder_input_ids)
        target_pos = positional_encodings(cast(torch.FloatTensor, target_tokens)).to(target_tokens.device)
        target_tokens = target_tokens + target_pos
        
        
        # TODO: To think about: Should query_features maintain some position encodings when
        #       used in the first encoder? Bert already adds them before feeding them to the first layer
        
        # Obtain query-aware frame representations
        qa_frames = self.feature_decoder(
            tgt=video_features,
            memory=text_features,
            tgt_key_padding_mask=~video_attention_mask.to(torch.bool), 
            memory_key_padding_mask=~text_attention_mask.to(torch.bool), 
        )
        
        # TODO: Shift decoder input ids by one to the right
        
        tgt_attn_mask = Transformer.generate_square_subsequent_mask(decoder_input_ids.shape[1]).to(decoder_input_ids.device)
        # re-add positions to qa_frames ?
        decoder_output = self.moment_decoder(
            tgt=target_tokens,
            memory=qa_frames,
            memory_key_padding_mask=~video_attention_mask.to(torch.bool),
            tgt_mask=tgt_attn_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=~decoder_attention_mask.to(torch.bool),
        )
        
        lm_logits = self.seq_modeling_head(decoder_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id_)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        if not return_dict:
            output = (lm_logits,) 
            return ((loss,) + output) if loss is not None else output
        
        return DoubleDecoderOutput(
            loss=loss,
            logits=lm_logits
        )
         


if __name__ == "__main__":
    # model = timm.create_model("resnet50", pretrained=True)
    # image_transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file("video_moment_retrieval/testing/tokenizer.json"),
        pad_token="[PAD]",
        unk_token="[UNK]"
    )
    dataset = DDDataset("qvhighlights_features/highlight_train_release.jsonl", "qvhighlights_features/bert_features", "qvhighlights_features/resnet_features", target_tokenizer=tokenizer)
    dataloader = DataLoader(dataset, 32, collate_fn=dataset.collate)
    batch = next(iter(dataloader))
    config = DoubleDecoderConfig(len(tokenizer.vocab), tokenizer.pad_token_id)
    model = DoubleDecoder(config)
    print(count_parameters(model, depth=2))
    output = model(**batch)
    # print(output)
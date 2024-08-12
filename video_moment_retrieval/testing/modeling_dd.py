from transformers import PreTrainedModel
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
from video_moment_retrieval.testing.configuration_dd import DoubleDecoderConfig


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
            video_features_path = os.path.join(
                self.video_features_root, data_item["vid"]) + ".npz"
            text_features_path = os.path.join(
                self.text_features_root, str(data_item["qid"])) + ".npy"
            video_features = np.load(video_features_path)["features"]
            text_features = np.load(text_features_path)[:self.max_query_len]
        except Exception as e:
            print(data_item["qid"])
            raise e

        # TODO: normalize features?
        # TODO: Somehow use timestamps as positional encodings
        relevant_windows = data_item["relevant_windows"][:self.max_relevant_windows]
        output = self.target_tokenizer(
            str(relevant_windows), return_tensors="pt")
        output = {k: v.squeeze(0) for k, v in output.items()}

        labels = output["input_ids"][1:]
        decoder_input_ids = output["input_ids"][:-1]
        decoder_attention_mask = output["attention_mask"][:-1]
        # decoder_attention_mask.masked_fill_(decoder_input_ids == self.target_tokenizer.eos_token_id, 0)
        # decoder_input_ids.masked_fill_(decoder_input_ids == self.target_tokenizer.eos_token_id, self.target_tokenizer.pad_token_id)

        return {
            "input_embeds": (torch.tensor(video_features), torch.tensor(text_features)),
            "attention_mask": (torch.ones(video_features.shape[0], dtype=torch.int32), torch.ones(text_features.shape[0], dtype=torch.int32)),
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }

    @staticmethod
    def pad_sequence(sequences: list, pad_value: int, dtype: torch.dtype) -> torch.Tensor:
        max_len = max(len(sequence) for sequence in sequences)
        padded_sequences = torch.zeros(
            (len(sequences), max_len, *sequences[0].shape[1:]), dtype=dtype) + pad_value
        for i, sequence in enumerate(sequences):
            padded_sequences[i, :len(sequence)] = sequence
        return padded_sequences

    def collate(self, batch: list[dict]) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        return {
            "input_embeds": (
                DDDataset.pad_sequence([item["input_embeds"][0] for item in batch], 0, torch.float32),
                DDDataset.pad_sequence([item["input_embeds"][1] for item in batch], 0, torch.float32)
            ),
            "attention_mask": (
                DDDataset.pad_sequence([item["attention_mask"][0] for item in batch], 0, torch.int32),
                DDDataset.pad_sequence([item["attention_mask"][1] for item in batch], 0, torch.int32),
            ),
            "decoder_input_ids": DDDataset.pad_sequence([item["decoder_input_ids"] for item in batch], self.target_tokenizer.pad_token_id, torch.int32),
            "decoder_attention_mask": DDDataset.pad_sequence([item["decoder_attention_mask"] for item in batch], 0, torch.int32),
            "labels": DDDataset.pad_sequence([item["labels"] for item in batch], self.target_tokenizer.pad_token_id, torch.int64),
        }


@dataclass
class DoubleDecoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class DDEncoder(PreTrainedModel):
    config_class = DoubleDecoderConfig

    def __init__(self, config: DoubleDecoderConfig):
        super().__init__(config)
        self.config = config
        self.text_projection = nn.Sequential(
            nn.LayerNorm(self.config.text_features_dim),
            nn.Dropout(config.dropout),
            nn.Linear(self.config.text_features_dim, self.config.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        self.frame_projection = nn.Sequential(
            nn.GroupNorm(1, self.config.video_features_shape[0]),
            nn.Dropout(config.dropout),
            nn.Conv2d(
                self.config.video_features_shape[0],
                self.config.d_model,
                self.config.video_features_shape[1]
            ),
            nn.ReLU(),
            nn.Flatten(1),
            nn.LayerNorm(self.config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        
        self.encoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            batch_first=True
        ), config.num_layers)

    def forward(
        self,
        input_embeds: tuple[torch.Tensor, torch.Tensor],
        attention_mask: tuple[torch.Tensor, torch.Tensor],
    ):
        video_features, text_features = input_embeds
        video_attention_mask, text_attention_mask = attention_mask
        bs, sq_l, channels, ks, _ = video_features.shape
        video_features = video_features.reshape(bs * sq_l, channels, ks, ks)
        video_features = self.frame_projection(video_features)
        video_features = video_features.reshape(bs, sq_l, -1)
        video_pos = positional_encodings(
            cast(torch.FloatTensor, video_features)).to(video_features.device)
        video_features = video_features + video_pos

        text_features = self.text_projection(text_features)

        # Obtain query-aware frame representations
        qa_frames = self.encoder(
            tgt=video_features,
            memory=text_features,
            tgt_key_padding_mask=~video_attention_mask.to(torch.bool),
            memory_key_padding_mask=~text_attention_mask.to(torch.bool),
        )
        
        return qa_frames


class DDDecoder(PreTrainedModel):
    def __init__(self, config: DoubleDecoderConfig):
        super().__init__(config)
        self.config = config
        self.out_vocab = nn.Embedding(
            self.config.vocab_size, self.config.d_model, self.config.pad_token_id)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            batch_first=True
        ), config.num_layers)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        target_tokens = self.out_vocab(decoder_input_ids)
        target_pos = positional_encodings(
            cast(torch.FloatTensor, target_tokens)).to(target_tokens.device)
        target_tokens = target_tokens + target_pos
        target_tokens = self.dropout(target_tokens)

        tgt_attn_mask = Transformer.generate_square_subsequent_mask(
            decoder_input_ids.shape[1]).to(decoder_input_ids.device)

        decoder_output = self.decoder(
            tgt=target_tokens,
            memory=encoder_output,
            memory_key_padding_mask=~encoder_attention_mask.to(torch.bool),
            tgt_mask=tgt_attn_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=~decoder_attention_mask.to(torch.bool),
        )
        # decoder_output = 
        return self.dropout(decoder_output)


class DoubleDecoderModel(PreTrainedModel):
    config_class = DoubleDecoderConfig

    def __init__(self, config: DoubleDecoderConfig):
        super().__init__(config)
        self.config = config

        self.encoder = DDEncoder(config)
        self.decoder = DDDecoder(config)
        self.seq_modeling_head = nn.Linear(
            self.config.d_model, self.config.vocab_size
        )

        self.post_init()

    def forward(
        self,
        input_embeds: tuple[torch.Tensor, torch.Tensor],
        attention_mask: tuple[torch.Tensor, torch.Tensor],
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):

        encoder_output = self.encoder(input_embeds, attention_mask)
        encoder_attention_mask = attention_mask[0]
        
        decoder_output = self.decoder(
            encoder_output=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )

        lm_logits = self.seq_modeling_head(decoder_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,)
            return ((loss,) + output) if loss is not None else output

        return DoubleDecoderOutput(
            loss=loss,
            logits=lm_logits
        )

    # def prepare_inputs_for_generation(self):
    #     pass


if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(
            "video_moment_retrieval/testing/tokenizer.json"),
        pad_token="[PAD]",
        unk_token="[UNK]"
    )
    dataset = DDDataset("qvhighlights_features/highlight_train_release.jsonl", "qvhighlights_features/bert_features",
                        "qvhighlights_features/resnet_features", target_tokenizer=tokenizer)
    dataloader = DataLoader(dataset, 32, collate_fn=dataset.collate)
    batch = next(iter(dataloader))
    config = DoubleDecoderConfig(len(tokenizer.vocab), tokenizer.pad_token_id)
    model = DoubleDecoderModel(config)
    print(count_parameters(model, depth=2))
    # model.generate(
    #     num_beams=4,
    #     do_sample=False,
    #     **batch
    # )
    output = model(**batch)

    # print(output)

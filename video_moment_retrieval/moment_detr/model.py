import torch
from transformers import DetrConfig
from transformers.models.detr.modeling_detr import DetrEncoder, DetrDecoder, ModelOutput
import torch.nn as nn
from typing import Optional
from transformers.modeling_utils import PreTrainedModel
from video_moment_retrieval.detr_matcher.matcher import VideoDetrHungarianMatcher, VideoDetrLoss
from video_moment_retrieval.datasets.qv_highlights import QVDataset, pad_collate
from torch.utils.data import DataLoader
from dataclasses import dataclass
from video_moment_retrieval.utils.logging import logger, init_logging
import pprint
from video_moment_retrieval.utils.utils import count_parameters


def positional_encodings(input_embs: torch.FloatTensor, n=10_000):
    batch_size, seq_len, d = input_embs.shape
    position = torch.arange(0, seq_len).unsqueeze_(1)
    denominator = torch.pow(n, 2 * torch.arange(0, d//2) / d)

    pos_enc = position / denominator
    encodings = torch.zeros((seq_len, d))
    encodings[:, 0::2] = pos_enc.sin()
    encodings[:, 1::2] = pos_enc.cos()
    return encodings.unsqueeze(0).repeat(batch_size, 1, 1)


@dataclass
class VideoDetrModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predicted_moments: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, hidden_dim: int, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(max_seq_len, hidden_dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bs, seq_l, _ = x.shape
        encondings = (self.embeddings.weight[None, :, :].repeat(bs, 1, 1))[
            :, :seq_l, :]
        return self.dropout(encondings)


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        h = [hidden_dim] * (n_layers - 1)
        for idx, (in_dim, out_dim) in enumerate(zip([input_dim] + h, h + [output_dim])):
            self.layers.extend([nn.LayerNorm(normalized_shape=in_dim),
                                nn.Dropout(dropout),
                                nn.Linear(in_dim, out_dim)])
            if idx < n_layers - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for layer in self.layers:
            x = layer(x)
        return x


class VideoDetrConfig(DetrConfig):
    def __init__(
        self,
        text_embedding_dim=512,
        video_embedding_dim=512,
        ce_loss_coefficient=4,
        saliency_loss_coefficient=1,
        hinge_loss_margin=0.2,
        **kwargs,
    ):
        # Input features
        self.text_embedding_dim = text_embedding_dim
        self.video_embedding_dim = video_embedding_dim
        self.ce_loss_coefficient = ce_loss_coefficient
        self.saliency_loss_coefficient = saliency_loss_coefficient
        self.hinge_loss_margin = hinge_loss_margin
        super().__init__(**kwargs)


class MomentDetr(PreTrainedModel):
    def __init__(self, config: VideoDetrConfig):
        super().__init__(config)

        self.config = config

        self.encoder = DetrEncoder(config)
        self.decoder = DetrDecoder(config)

        self.text_projection = ProjectionMLP(
            config.text_embedding_dim, config.hidden_size, config.hidden_size, dropout=0.5)
        self.video_projection = ProjectionMLP(
            config.video_embedding_dim, config.hidden_size, config.hidden_size, dropout=0.5)

        self.object_queries = nn.Embedding(config.num_queries, config.d_model)

        self.sal_predictor = nn.Linear(config.hidden_size, 1)

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

    def forward(
        self,
        video_features: torch.FloatTensor,
        video_attn_mask: Optional[torch.FloatTensor],
        text_features: torch.FloatTensor,
        text_attn_mask: Optional[torch.FloatTensor],
        labels: Optional[list[dict[str, torch.Tensor]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> torch.FloatTensor:
        batch_size, video_seq_len, _ = video_features.shape
        _, text_seq_len, _ = text_features.shape

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if video_attn_mask is None:
            video_attn_mask = torch.ones(
                (batch_size, video_seq_len), device=video_features.device)
        if text_attn_mask is None:
            text_attn_mask = torch.ones(
                (batch_size, text_seq_len), device=text_features.device)

        text_projected = self.text_projection(text_features)
        video_projected = self.video_projection(video_features)

        features = torch.cat([video_projected, text_projected], dim=1)
        attn_mask = torch.cat([video_attn_mask, text_attn_mask], dim=1)

        # Compute positional encodings
        vid_postions = positional_encodings(video_projected).to(self.device)
        text_positions = torch.zeros_like(text_projected, device=self.device)

        positions = torch.cat([vid_postions, text_positions], dim=1)

        # Pass through the encoder using positions and concatenated_features
        encoder_output = self.encoder(
            inputs_embeds=features,
            attention_mask=attn_mask,
            object_queries=positions,
        )

        # N x L
        saliency_output = torch.squeeze(self.sal_predictor(
            encoder_output.last_hidden_state[:, :video_seq_len]), dim=-1)

        # Pass through the decoder using positions, object_queries, encoder_output
        object_queries = self.object_queries.weight.unsqueeze(
            0).repeat(batch_size, 1, 1)
        decoder_inputs = torch.zeros_like(object_queries)

        decoder_output = self.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=None,
            encoder_hidden_states=encoder_output.last_hidden_state,
            encoder_attention_mask=attn_mask,
            object_queries=positions,
            query_position_embeddings=object_queries
        )

        pred_moments = self.moment_predictor(
            decoder_output.intermediate_hidden_states)
        logits = self.classifier(decoder_output.intermediate_hidden_states)
        aux_outputs = None
        if self.config.auxiliary_loss:
            aux_outputs = [
                {
                    "logits": logits_,
                    "pred_boxes": pred_moments_
                } for (logits_, pred_moments_) in zip(logits[:-1], pred_moments[:-1])
            ]

        loss = None
        if labels is not None:
            outputs_loss = {
                "logits": logits[-1],
                "pred_boxes": pred_moments[-1]
            }
            if aux_outputs:
                outputs_loss["auxiliary_outputs"] = aux_outputs
            matcher = VideoDetrHungarianMatcher(
                self.config.class_cost, self.config.bbox_cost, self.config.giou_cost)
            criterion = VideoDetrLoss(
                matcher, self.config.num_labels, self.config.eos_coefficient, ["labels", "boxes"])
            criterion.to(self.device)

            loss_dict = criterion(outputs_loss, labels)

            positive_values = torch.stack(
                [saliency_output[idx, label["positive_ids"]] for idx, label in enumerate(labels)])
            negative_values = torch.stack(
                [saliency_output[idx, label["negative_ids"]] for idx, label in enumerate(labels)])

            sal_loss = torch.clamp(
                self.config.hinge_loss_margin + negative_values - positive_values, min=0)
            sal_loss = sal_loss.sum() / (len(labels) *
                                         positive_values.shape[1]) * 2

            loss_dict["loss_saliency"] = sal_loss
            weight_dict = {
                "loss_ce": self.config.ce_loss_coefficient,
                "loss_bbox": self.config.bbox_loss_coefficient,
                "loss_giou": self.config.giou_loss_coefficient,
                "loss_saliency": self.config.saliency_loss_coefficient
            }
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
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
    init_logging()
    train_dataset = QVDataset("qvhighlights_features\\text_features",
                              "qvhighlights_features\\video_features", "qvhighlights_features\\highlight_train_release.jsonl")
    eval_dataset = QVDataset("qvhighlights_features\\text_features",
                             "qvhighlights_features\\video_features", "qvhighlights_features\\highlight_val_release.jsonl")

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
    logger.info("Instantiating model using config %s", config)

    model = MomentDetr(config)

    params_count = count_parameters(model)

    logger.info(pprint.pprint(params_count))

    train_loader = DataLoader(train_dataset, 32, False, collate_fn=pad_collate)

    batch = next(iter(train_loader))

    output = model(**batch)

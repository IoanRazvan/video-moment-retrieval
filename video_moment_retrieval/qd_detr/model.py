from transformers import PreTrainedModel, DetrConfig
from transformers.models.detr.modeling_detr import DetrEncoder
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as f
from video_moment_retrieval.moment_detr.model import positional_encodings


class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float, layer_norm_eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.ca_layer = nn.MultiheadAttention(
            d_model, n_heads, dropout, bias=bias, batch_first=True)
        self.norm_1 = nn.LayerNorm(d_model, layer_norm_eps, bias=bias)
        self.norm_2 = nn.LayerNorm(d_model, layer_norm_eps, bias=bias)
        self.linear_1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.linear_2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, video_features: torch.FloatTensor, text_features: torch.FloatTensor, text_mask: torch.FloatTensor) -> torch.FloatTensor:
        x = video_features
        x = self.norm_1(x + self.dropout_1(self.ca_layer(x,
                        text_features, text_features, key_padding_mask=text_mask)[0]))
        x = self.norm_2(x + self._ffn(x))
        return x

    def _ffn(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.linear_2(self.dropout_2(f.relu(self.linear_1(x))))
        return self.dropout_3(x)


class CrossAttentionEncoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, dim_feedforward: int, dropout: float, layer_norm_eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.encoder = nn.ModuleList([
            CrossAttentionEncoderLayer(d_model, n_heads, dim_feedforward, dropout, layer_norm_eps, bias)
            for _ in range(n_layers)
        ])

    def forward(self, video_features: torch.FloatTensor, text_features: torch.FloatTensor, text_mask: torch.FloatTensor) -> torch.FloatTensor:
        output = video_features
        for layer in self.encoder:
            output = layer(output, text_features, text_mask)
        return output
    
class DABDetrDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    
class QDDetr(PreTrainedModel):
    def __init__(self, config: DetrConfig):
        super().__init__(config)
        self.config = config
        self.saliency_token = nn.Embedding(1, config.d_model)
        self.ca_encoder = CrossAttentionEncoder(
            config.encoder_layers, config.d_model, config.encoder_attention_heads, config.encoder_ffn_dim, dropout=config.dropout)
        self.detr_encoder = DetrEncoder(config)
        self.sal_token_projection = nn.Linear(config.d_model, 1)
        self.vid_tokens_projection = nn.Linear(config.d_model, 1)
        self.moment_queries = nn.Embedding(config.num_queries, config.query_dim)
        self.post_init()

    def forward(self,
                video_features: torch.FloatTensor,
                video_attn_mask: Optional[torch.FloatTensor],
                text_features: torch.FloatTensor,
                text_attn_mask: Optional[torch.FloatTensor],
                labels: Optional[list[dict[str, torch.Tensor]]] = None,
                return_dict: Optional[bool] = None,
                **kwargs
                ) -> torch.FloatTensor:
        # Step 1: Apply cross attention on video_features and text_features
        # qd_video_tokens: bs x vid_len x embedd_dim
        bs, _, _ = video_features.shape
        # TODO: Add positional encodings to video_features?
        qd_video_tokens = self.ca_encoder(video_features, text_features, text_attn_mask)

        saliency_token = self.saliency_token.weight[:, None, :].repeat((bs, 1, 1))
        # encoder_input: bs x vid_len + 1 x embedd_dim
        encoder_input = torch.cat([saliency_token, qd_video_tokens], dim=1)
        
        # video_attn_mask: bs x vid_len
        # encoder_attn_mask: bs x vid_len + 1
        encoder_attn_mask = torch.cat([torch.ones((bs, 1), device=self.device), video_attn_mask], dim=1)
        pos_encodings = positional_encodings(qd_video_tokens)
        
        # Step 2: Apply DetrEncoder on the results from step 1
        encoder_output = self.detr_encoder(encoder_input, encoder_attn_mask, pos_encodings)
        
        # Step 3: Compute saliency scores
        # shape: bs x 1 x 1 or bs x 1
        projected_saliency_token = self.sal_token_projection(encoder_output.last_hidden_state[:, 0, :])
        # shape: bs x vid_len x 1
        projected_vid_tokens = self.vid_tokens_projection(encoder_output.last_hidden_state[:, 1:, :])
        # saliency_scores: bs x vid_len
        saliency_scores = (projected_saliency_token * projected_vid_tokens) / torch.sqrt(self.config.d_model)
        
        # Step 4: Call DAB-DETR Decoder on the results from step 2
        
        # Step 5: Find assignment between moment queries and g.t. moments and compute loss
        pass
from transformers import PreTrainedModel, DetrConfig
import torch
import torch.nn as nn
from typing import Optional

class QDDetr(PreTrainedModel):
    def __init__(self, config: DetrConfig):
        self.saliency_token = nn.Embedding(1, config.d_model)
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
        
        # Step 2: Apply DetrEncoder on the results from step 1
        
        # Step 3: Append Saliency token to result from step 2
        
        # Step 4: Call DAB-DETR Decoder on the results from step 3
        
        pass
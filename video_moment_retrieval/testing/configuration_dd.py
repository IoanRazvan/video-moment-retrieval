from transformers import PretrainedConfig

class DoubleDecoderConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 97,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        is_encoder_decoder: bool = True,
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
        self.pad_token_id = pad_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs
        )


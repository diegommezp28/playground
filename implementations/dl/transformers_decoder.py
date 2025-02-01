import torch


class SelfAttention(torch.nn.Module):
    pass


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Decoder layer of the transformer.

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of heads in the multi-head attention.
            d_ff (int): Dimension of latent space in the feed-forward layer.
            dropout (float): Dropout rate.

        """

        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = SelfAttention(d_model, num_heads, dropout)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model),
        )
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass of the decoder layer.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        x = self.layer_norm1(
            x + self.dropout(self.self_attention(x, x, x, mask))
        )
        x = self.layer_norm2(x + self.dropout(self.feed_forward(x)))
        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        """
        Decoder style transformer. (BERT-like) without masked self-attention.

        Args:
            num_layers (int): Number of layers in the transformer.
            d_model (int): Dimension of the model.
            num_heads (int): Number of heads in the multi-head attention.
            d_ff (int): Dimension of latent space in the feed-forward layer.
            dropout (float): Dropout rate.

        """

        super(TransformerDecoder, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

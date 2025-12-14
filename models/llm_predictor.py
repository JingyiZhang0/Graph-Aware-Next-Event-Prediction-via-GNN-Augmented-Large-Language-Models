# Title: llm_predictor.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: models/llm_predictor.py
# Description: LLM-based classifier that fuses GNN embeddings as a special token.

import torch.nn as nn
import torch
    
class FusionLLMClassifier(nn.Module):
    def __init__(
        self,
        llm_model: nn.Module,
        gnn_emb_dim: int,
        llm_embed_dim: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        """FusionLLMClassifier

        A classifier that prepends a projected GNN embedding as a special token
        to the LLM input embeddings and uses the first token output for classification.

        Args:
            llm_model (transformers.PreTrainedModel): Encoder-only LLM backbone.
            gnn_emb_dim (int): Dimension of incoming GNN sequence embedding.
            llm_embed_dim (int): LLM hidden size / embedding dimension.
            num_labels (int): Number of classification labels.
            dropout (float): Dropout probability before final linear layer.

        Attributes:
            llm (transformers.PreTrainedModel): Backbone encoder model.
            project_gnn (torch.nn.Linear): Projects GNN embedding to `llm_embed_dim`.
            classifier (torch.nn.Sequential): Dropout + Linear to logits of size `num_labels`.
        """
        super().__init__()
        self.llm = llm_model  # e.g., DistilBertModel
        self.project_gnn = nn.Linear(gnn_emb_dim, llm_embed_dim)  # Map GNN embeddings to the same dimension as LLM embeddings
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # Add dropout layer
            nn.Linear(llm_embed_dim, num_labels)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gnn_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids (torch.Tensor): Shape [bs, seq_len], integer token ids.
            attention_mask (torch.Tensor): Shape [bs, seq_len], attention mask (0/1).
            gnn_emb (torch.Tensor): Shape [bs, gnn_emb_dim], structural sequence embedding.

        Returns:
            torch.Tensor: Logits of shape [bs, num_labels].

        Processing steps:
            1) Convert `input_ids` to token embeddings.
            2) Project `gnn_emb` to `llm_embed_dim` as a special token embedding.
            3) Prepend the special token to the token embeddings.
            4) Extend `attention_mask` with 1 for the special token.
            5) Run the LLM encoder with `inputs_embeds` and `attention_mask`.
            6) Use the first token output for classification.
        """

        bs, seq_len = input_ids.size()
        device = input_ids.device

        #
        # Use the unified HF API to obtain input embeddings for any encoder-only model
        input_embedding_layer = self.llm.get_input_embeddings()
        token_embeds = input_embedding_layer(input_ids)  # [bs, seq_len, llm_embed_dim]

        # Map gnn_emb
        gnn_token_emb = self.project_gnn(gnn_emb).unsqueeze(1)  # [bs, 1, llm_embed_dim]

        # Concatenate
        fused_embeds = torch.cat([gnn_token_emb, token_embeds], dim=1)  # [bs, seq_len+1, llm_embed_dim]

        # New attention_mask
        extended_attention_mask = torch.cat(
            [torch.ones((bs, 1), device=device, dtype=attention_mask.dtype), attention_mask], dim=1
        )  # [bs, seq_len+1]

        # Call LLM encoder (model-agnostic forward)
        outputs = self.llm(
            inputs_embeds=fused_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state  # [bs, seq_len+1, llm_embed_dim]

        # Classification (use the output corresponding to the first token, i.e., gnn_token_emb)
        cls_token_state = last_hidden_state[:, 0, :]  # [bs, llm_embed_dim]
        logits = self.classifier(cls_token_state)  # [bs, num_labels]

        return logits
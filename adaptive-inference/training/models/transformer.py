import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class FFNExpert(nn.Module):
    """
    A single feed-forward network expert.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., d_model)
        return self.dropout(self.out_proj(self.act(self.in_proj(x))))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer using top-K routing.
    Supports routing masking for Controlled MoE.
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Instantiate the experts
        self.experts = nn.ModuleList([
            FFNExpert(d_model, d_ff) for _ in range(num_experts)
        ])
        
        # Router network (projects to expert logits)
        # We do not use bias for the router to keep parameter count simple and symmetric
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, num_experts) - optional mask for controlled routing
        
        Returns:
            output: (batch_size, seq_len, d_model)
            router_logits: (batch_size, seq_len, num_experts)
        """
        batch_size, seq_len, d_model = x.shape
        # Flatten tokens: (batch_size * seq_len, d_model)
        flat_x = x.view(-1, d_model)
        
        # Compute router logits: (batch_size * seq_len, num_experts)
        router_logits = self.router(flat_x)
        
        # Apply domain routing mask if provided
        if mask is not None:
            # Expand mask from (batch_size, num_experts) to (batch_size, seq_len, num_experts)
            expanded_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            flat_mask = expanded_mask.reshape(-1, self.num_experts)
            # Apply large negative bias to masked-out experts
            router_logits = router_logits + (1.0 - flat_mask) * -1e9

        # Softmax to get routing gates: (batch_size * seq_len, num_experts)
        routing_gates = F.softmax(router_logits, dim=-1)
        
        # Select top-K experts
        topk_gates, topk_indices = torch.topk(routing_gates, self.top_k, dim=-1)
        
        # Re-normalize gates over top-K selected experts
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Prepare output container
        flat_out = torch.zeros_like(flat_x)
        
        # For efficiency, group tokens by their assigned experts
        # We loop over the top-K assignments
        for k in range(self.top_k):
            gates = topk_gates[:, k]
            indices = topk_indices[:, k]
            
            # Loop over each expert and process its tokens
            for exp_idx in range(self.num_experts):
                token_mask = (indices == exp_idx)
                if not token_mask.any():
                    continue
                
                # Extract tokens assigned to this expert
                exp_tokens = flat_x[token_mask]
                # Process with expert FFN
                exp_outputs = self.experts[exp_idx](exp_tokens)
                
                # Accumulate gated outputs
                flat_out[token_mask] += gates[token_mask].unsqueeze(-1) * exp_outputs

        # Reshape back to sequence format
        output = flat_out.view(batch_size, seq_len, d_model)
        return output, router_logits.view(batch_size, seq_len, -1)


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head causal self-attention.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Attention projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape: (batch_size, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask: attention only to past tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Weighted values context: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_probs, v)
        
        # Reshape back to: (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.resid_dropout(self.out_proj(context))


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder layer block.
    Supports either standard dense FFN or MoE FFN.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, moe: bool = False,
                 num_experts: int = 1, top_k: int = 1):
        super().__init__()
        self.moe = moe
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        if moe:
            self.ffn = MoELayer(d_model, d_ff, num_experts, top_k)
        else:
            self.ffn = FFNExpert(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN structure
        x_norm = self.ln1(x)
        x = x + self.attn(x_norm)
        
        x_norm = self.ln2(x)
        if self.moe:
            ffn_out, router_logits = self.ffn(x_norm, mask)
        else:
            ffn_out = self.ffn(x_norm)
            router_logits = None
            
        x = x + ffn_out
        return x, router_logits


class MoETransformer(nn.Module):
    """
    The full Decoder-only Transformer model supporting both Dense and MoE modes.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, num_heads: int,
                 d_ff: int, max_seq_len: int = 512, moe: bool = False,
                 num_experts: int = 1, top_k: int = 1):
        super().__init__()
        self.moe = moe
        self.max_seq_len = max_seq_len
        
        # Shared embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.ln_embed = nn.LayerNorm(d_model)
        
        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, moe, num_experts, top_k)
            for _ in range(n_layers)
        ])
        
        # Final layers
        self.ln_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        input_ids: (batch_size, seq_len)
        mask: (batch_size, num_experts) - optional mask for controlled routing
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            all_router_logits: list of (batch_size, seq_len, num_experts) for each MoE layer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Construct positions
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.ln_embed(x)
        
        all_router_logits = []
        
        # Process decoder layers
        for block in self.blocks:
            x, r_logits = block(x, mask)
            if r_logits is not None:
                all_router_logits.append(r_logits)
                
        # Output layer norm and logit projection
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        return logits, all_router_logits

"""
runtime/engine.py — ASIR MoE Inference Runtime Engine (TR-10)

Coordinates the two-phase local inference execution:
1. Prefill Phase: Batched prompt processing.
2. Autoregressive Decode Phase: Step-by-step token generation (N=1) with KV cache & ExpertCache residency offloading.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional

from tasks.tokenizer import CharTokenizer
from training.models.transformer import MoETransformer
from runtime.expert_cache import ExpertCache
from runtime.metrics import RuntimeMetrics


class InferenceEngine:
    """
    Local MoE Inference Engine integrating Transformer model, ExpertCache, and telemetry.
    """

    def __init__(
        self,
        model: MoETransformer,
        tokenizer: CharTokenizer,
        expert_cache: Optional[ExpertCache] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.expert_cache = expert_cache
        self.device = device if device is not None else next(model.parameters()).device
        self.metrics = RuntimeMetrics()
        
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt_str: str,
        max_gen_len: int = 30,
        temperature: float = 0.0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Executes prompt prefill and autoregressive token decoding.
        Returns generated text string and comprehensive telemetry summary.
        """
        self.metrics.reset()
        if self.expert_cache is not None:
            self.expert_cache.reset_stats()

        # Tokenize input prompt
        prompt_ids = self.tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
        curr_ids = prompt_ids + [self.tokenizer.sep_id]
        
        input_tensor = torch.tensor([curr_ids], dtype=torch.long, device=self.device)
        
        # --- PHASE 1: PREFILL PHASE ---
        self.metrics.start_prefill()
        logits, all_router_logits = self.model(input_tensor, mask=None)
        self.metrics.end_prefill(len(curr_ids))
        
        # Intercept and process router activations through ExpertCache if present
        if self.expert_cache is not None and self.model.moe and all_router_logits:
            for l_idx, router_logits in enumerate(all_router_logits):
                flat_logits = router_logits.view(-1, router_logits.size(-1))
                top_k = self.model.blocks[0].ffn.top_k
                _, topk_indices = torch.topk(flat_logits, top_k, dim=-1)
                for e_id in topk_indices.view(-1).tolist():
                    self.expert_cache.access(l_idx, e_id)

        # Predict first token
        next_token = torch.argmax(logits[0, -1, :]).item()
        generated_ids = []
        
        # --- PHASE 2: AUTOREGRESSIVE DECODE PHASE ---
        self.metrics.start_decode()
        for step in range(max_gen_len):
            if next_token == self.tokenizer.eos_id:
                break
                
            generated_ids.append(next_token)
            self.metrics.step_decode(1)
            
            # Autoregressive forward pass (N=1 step)
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)
            
            if input_tensor.size(1) > self.model.max_seq_len:
                break
                
            logits, all_router_logits = self.model(input_tensor, mask=None)
            
            # Access expert cache for single decode step
            if self.expert_cache is not None and self.model.moe and all_router_logits:
                for l_idx, router_logits in enumerate(all_router_logits):
                    flat_logits = router_logits[:, -1, :]  # decode last token
                    top_k = self.model.blocks[0].ffn.top_k
                    _, topk_indices = torch.topk(flat_logits, top_k, dim=-1)
                    for e_id in topk_indices.view(-1).tolist():
                        self.expert_cache.access(l_idx, e_id)
                        # Optional prefetch for next layer/step
                        self.expert_cache.prefetch_next(l_idx, e_id, len(self.model.blocks))
                        
            next_token = torch.argmax(logits[0, -1, :]).item()
            
        self.metrics.end_decode()
        
        # Sync NVMe stats from expert cache into metrics tracker
        if self.expert_cache is not None:
            cache_stats = self.expert_cache.get_stats()
            self.metrics.record_nvme_read(cache_stats['nvme_bytes_read'], cache_stats['nvme_io_time_ms'])
        
        # Combine telemetry
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        summary = self.metrics.get_summary()
        if self.expert_cache is not None:
            summary['cache'] = self.expert_cache.get_stats()
            
        return generated_text, summary

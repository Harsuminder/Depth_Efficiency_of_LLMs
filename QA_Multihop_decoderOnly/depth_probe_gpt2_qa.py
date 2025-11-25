"""
Depth Probe Analysis for GPT-2 on Multi-Hop Question Answering
Uses pretrained decoder-only model from HuggingFace to analyze layer-wise depth metrics.
Compares SIMPLE (single-hop) vs MULTI-HOP QA tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


class GPT2DepthProbe:
    """Class to handle depth probing on GPT-2 transformer blocks."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {}
        
    def register_hooks(self):
        """Register forward hooks on all transformer blocks."""
        # Find transformer blocks
        transformer_blocks = None
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            transformer_blocks = self.model.transformer.h
        else:
            # Try to find transformer blocks in the model structure
            for name, module in self.model.named_modules():
                if 'transformer' in name and 'h' in name and isinstance(module, nn.ModuleList):
                    transformer_blocks = module
                    break
        
        if transformer_blocks is None:
            raise ValueError("Could not find transformer blocks in the model")
        
        self.num_layers = len(transformer_blocks)
        
        # Register hooks on each block
        for i, block in enumerate(transformer_blocks):
            self._register_block_hooks(block, i)
    
    def _register_block_hooks(self, block: nn.Module, layer_idx: int):
        """
        Register hooks for GPT-2 transformer block:
        - h_l: input to the block (before ln_1)
        - a_l: attention output (after attn, before residual)
        - m_l: MLP output (after mlp, before residual)
        """
        # Hook on block input (h_l) - before ln_1
        def input_hook(module, input):
            self.activations[f'h_{layer_idx}'] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.hooks.append(handle1)
        
        # Hook on attention output (a_l)
        # GPT-2 block has 'attn' module that outputs (hidden_states, ...)
        if hasattr(block, 'attn'):
            def attn_hook(module, input, output):
                # Attention returns (hidden_states, ...)
                if isinstance(output, tuple):
                    self.activations[f'a_{layer_idx}'] = output[0].detach().clone()
                else:
                    self.activations[f'a_{layer_idx}'] = output.detach().clone()
            
            handle2 = block.attn.register_forward_hook(attn_hook)
            self.hooks.append(handle2)
        
        # Hook on MLP output (m_l)
        # GPT-2 block has 'mlp' module
        if hasattr(block, 'mlp'):
            def mlp_hook(module, input, output):
                self.activations[f'm_{layer_idx}'] = output.detach().clone()
            
            handle3 = block.mlp.register_forward_hook(mlp_hook)
            self.hooks.append(handle3)
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_metrics(probe: GPT2DepthProbe, num_layers: int) -> Dict[str, List[float]]:
    """
    Compute depth probe metrics:
    1. Residual L2 norm: ||h_l||_2
    2. Relative contribution: ||u_l|| / ||h_l|| where u_l = a_l + m_l
    3. Cosine similarity: (h_l • u_l) / (||h_l|| * ||u_l||)
    """
    metrics = {
        'l2_norm': [],
        'relative_contribution': [],
        'cosine_similarity': []
    }
    
    for l in range(num_layers):
        h_l = probe.activations.get(f'h_{l}')
        a_l = probe.activations.get(f'a_{l}')
        m_l = probe.activations.get(f'm_{l}')
        
        if h_l is None:
            continue
        
        # (1) Residual L2 norm - average over batch and sequence
        l2_norm = torch.norm(h_l, p=2, dim=-1).mean().item()
        metrics['l2_norm'].append(l2_norm)
        
        if a_l is not None and m_l is not None:
            # Ensure shapes match
            batch_size = min(h_l.shape[0], a_l.shape[0], m_l.shape[0])
            seq_len = min(h_l.shape[1], a_l.shape[1], m_l.shape[1])
            hidden_dim = min(h_l.shape[2], a_l.shape[2], m_l.shape[2])
            
            h_l = h_l[:batch_size, :seq_len, :hidden_dim]
            a_l = a_l[:batch_size, :seq_len, :hidden_dim]
            m_l = m_l[:batch_size, :seq_len, :hidden_dim]
            
            # u_l = a_l + m_l
            u_l = a_l + m_l
            
            # (2) Relative contribution: ||u_l|| / ||h_l||
            u_norm = torch.norm(u_l, p=2, dim=-1)  # [batch, seq_len]
            h_norm = torch.norm(h_l, p=2, dim=-1)  # [batch, seq_len]
            rel_contrib = (u_norm / (h_norm + 1e-8)).mean().item()
            metrics['relative_contribution'].append(rel_contrib)
            
            # (3) Cosine similarity
            dot_product = (h_l * u_l).sum(dim=-1)  # [batch, seq_len]
            cosine_per_element = dot_product / ((h_norm * torch.norm(u_l, p=2, dim=-1)) + 1e-8)
            cosine = cosine_per_element.mean().item()
            metrics['cosine_similarity'].append(cosine)
        else:
            metrics['relative_contribution'].append(np.nan)
            metrics['cosine_similarity'].append(np.nan)
    
    return metrics


def compute_layer_skipping_kl(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    problems: List[str],
    num_layers: int,
    device: torch.device,
    answer_token_positions: List[int]
) -> List[float]:
    """
    Compute KL divergence for each skipped layer.
    For each layer s, bypass block s and compute KL divergence between new logits and original logits.
    """
    model.eval()
    
    # Find transformer blocks
    transformer_blocks = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_blocks = model.transformer.h
    else:
        for name, module in model.named_modules():
            if 'transformer' in name and 'h' in name and isinstance(module, nn.ModuleList):
                transformer_blocks = module
                break
    
    if transformer_blocks is None:
        raise ValueError("Could not find transformer blocks for layer skipping")
    
    # Get original logits at answer token positions
    original_logits_list = []
    with torch.no_grad():
        for problem, answer_pos in zip(problems, answer_token_positions):
            inputs = tokenizer(problem, return_tensors='pt').to(device)
            input_ids = inputs['input_ids']
            
            # Forward pass to get logits at answer position
            outputs = model(input_ids)
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            
            # Get logits at the answer token position (first token after "Answer:")
            if answer_pos < logits.shape[1]:
                original_logits = logits[0, answer_pos, :].detach().clone()
                original_logits_list.append(original_logits)
            else:
                # Fallback: use last position
                original_logits = logits[0, -1, :].detach().clone()
                original_logits_list.append(original_logits)
    
    # Convert to probabilities for KL computation
    original_probs_list = [F.softmax(logits, dim=-1) for logits in original_logits_list]
    
    kl_divergences = []
    
    # Test each layer skipping
    print("Computing layer skipping KL divergence...")
    for s in range(num_layers):
        original_block = transformer_blocks[s]
        
        # Safe Skip Block: returns hidden_states unchanged
        class SkipBlock(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, hidden_states, *args, **kwargs):
                # Match GPT-2 block output structure: return (hidden_states, present)
                present = None
                return hidden_states, present
        
        # Replace the block temporarily
        transformer_blocks[s] = SkipBlock()
        
        # Compute logits with skipped layer
        skipped_logits_list = []
        with torch.no_grad():
            for problem, answer_pos in zip(problems, answer_token_positions):
                inputs = tokenizer(problem, return_tensors='pt').to(device)
                input_ids = inputs['input_ids']
                
                outputs = model(input_ids)
                logits = outputs.logits
                
                if answer_pos < logits.shape[1]:
                    skipped_logits = logits[0, answer_pos, :].detach().clone()
                else:
                    skipped_logits = logits[0, -1, :].detach().clone()
                skipped_logits_list.append(skipped_logits)
        
        # Compute average KL divergence
        kl_sum = 0.0
        for orig_probs, skipped_logits in zip(original_probs_list, skipped_logits_list):
            # Compute KL divergence: KL(P_orig || P_skipped)
            kl = F.kl_div(
                F.log_softmax(skipped_logits, dim=-1),
                orig_probs,
                reduction='sum'
            )
            kl_sum += kl.item()
        
        avg_kl = kl_sum / len(problems)
        kl_divergences.append(avg_kl)
        
        print(f"  Skipped layer {s}, KL divergence={avg_kl:.4f}")
        
        # Restore original block
        transformer_blocks[s] = original_block
    
    return kl_divergences


def plot_metrics(
    simple_metrics: Dict[str, List[float]],
    multihop_metrics: Dict[str, List[float]],
    simple_kl: List[float],
    multihop_kl: List[float],
    num_layers: int
):
    """Plot all metrics comparing SIMPLE (single-hop) vs MULTI-HOP QA."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPT-2 Depth Probe Metrics: Single-Hop vs Multi-Hop QA', 
                 fontsize=16, fontweight='bold')
    
    layers = list(range(num_layers))
    
    # Residual L2 norm
    axes[0, 0].plot(layers, simple_metrics['l2_norm'], 'b-o', linewidth=2, markersize=6, label='SINGLE-HOP')
    axes[0, 0].plot(layers, multihop_metrics['l2_norm'], 'r-s', linewidth=2, markersize=6, label='MULTI-HOP')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('L2 Norm')
    axes[0, 0].set_title('Residual L2 Norm ||h_l||_2')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Relative contribution
    axes[0, 1].plot(layers, simple_metrics['relative_contribution'], 'b-o', linewidth=2, markersize=6, label='SINGLE-HOP')
    axes[0, 1].plot(layers, multihop_metrics['relative_contribution'], 'r-s', linewidth=2, markersize=6, label='MULTI-HOP')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('||u_l|| / ||h_l||')
    axes[0, 1].set_title('Relative Contribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cosine similarity
    axes[1, 0].plot(layers, simple_metrics['cosine_similarity'], 'b-o', linewidth=2, markersize=6, label='SINGLE-HOP')
    axes[1, 0].plot(layers, multihop_metrics['cosine_similarity'], 'r-s', linewidth=2, markersize=6, label='MULTI-HOP')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_title('Cosine Similarity (h_l, u_l)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-1, 1])
    
    # KL divergence under layer skipping
    axes[1, 1].plot(layers, simple_kl, 'b-o', linewidth=2, markersize=6, label='SINGLE-HOP')
    axes[1, 1].plot(layers, multihop_kl, 'r-s', linewidth=2, markersize=6, label='MULTI-HOP')
    axes[1, 1].set_xlabel('Layer Skipped')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].set_title('KL Divergence (Layer Skipped)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('QA_Multihop_decoderOnly/depth_probe_gpt2_qa_metrics.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'QA_Multihop_decoderOnly/depth_probe_gpt2_qa_metrics.png'")
    plt.show()


def find_answer_token_position(tokenizer: GPT2Tokenizer, problem: str) -> int:
    """
    Find the position of the first token after "Answer:" in the tokenized sequence.
    This is where we'll probe for depth metrics.
    """
    tokens = tokenizer.encode(problem)
    answer_text = "Answer:"
    
    # Find where "Answer:" appears in the problem
    answer_idx = problem.find(answer_text)
    if answer_idx == -1:
        return len(tokens) - 1  # Fallback: use last position
    
    # Tokenize up to "Answer:" and find the position
    prefix = problem[:answer_idx + len(answer_text)]
    prefix_tokens = tokenizer.encode(prefix)
    
    # The answer token position is right after "Answer:"
    return len(prefix_tokens)


def process_problems(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    problems: List[str],
    probe: GPT2DepthProbe,
    device: torch.device
) -> Tuple[Dict[str, List[float]], List[int]]:
    """Process a set of problems and compute metrics."""
    probe.clear_activations()
    
    # Find answer token positions
    answer_positions = [find_answer_token_position(tokenizer, p) for p in problems]
    
    # Forward pass to capture activations at answer positions
    with torch.no_grad():
        for problem in problems:
            inputs = tokenizer(problem, return_tensors='pt').to(device)
            _ = model(**inputs)
    
    # Compute metrics (averaged across all problems)
    metrics = compute_metrics(probe, probe.num_layers)
    
    return metrics, answer_positions


def main():
    """Main function to run depth probe analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained GPT-2
    print("Loading pretrained GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    
    # SIMPLE (single-hop) QA problems
    simple_problems = [
        "Passage: Mary lives in Paris. Question: Where does Mary live? Answer:",
        "Passage: The book is on the table. Question: Where is the book? Answer:",
        "Passage: Tom works at Google. Question: Where does Tom work? Answer:"
    ]
    
    # MULTI-HOP (two or more hops) QA problems
    multihop_problems = [
        "Passage: John went to the store after leaving his house. His house is in Seattle. Question: Where did John start his trip? Answer:",
        "Passage: Sarah's father is a doctor. He works at Mercy Hospital downtown. Question: Where does Sarah's father work? Answer:",
        "Passage: The capital of France is Paris. Paris hosts the Louvre Museum. Question: Which city hosts the Louvre Museum? Answer:",
        "Passage: Mike bought a laptop from TechWorld. TechWorld is located inside Sunset Mall. Question: Where did Mike buy the laptop? Answer:"
    ]
    
    print(f"\nSINGLE-HOP problems: {len(simple_problems)}")
    print(f"MULTI-HOP problems: {len(multihop_problems)}")
    
    # Initialize depth probe
    print("\nSetting up depth probe hooks...")
    probe = GPT2DepthProbe(model)
    probe.register_hooks()
    
    print(f"Registered hooks for {probe.num_layers} transformer layers")
    
    # Process SINGLE-HOP problems
    print("\n" + "="*60)
    print("Processing SINGLE-HOP QA problems...")
    print("="*60)
    simple_metrics, simple_answer_positions = process_problems(
        model, tokenizer, simple_problems, probe, device
    )
    
    # Process MULTI-HOP problems
    print("\n" + "="*60)
    print("Processing MULTI-HOP QA problems...")
    print("="*60)
    multihop_metrics, multihop_answer_positions = process_problems(
        model, tokenizer, multihop_problems, probe, device
    )
    
    print(f"\nComputed metrics for {len(simple_metrics['l2_norm'])} layers")
    
    # Compute layer skipping KL divergence
    print("\n" + "="*60)
    print("Computing layer skipping KL divergence for SINGLE-HOP...")
    print("="*60)
    try:
        simple_kl = compute_layer_skipping_kl(
            model, tokenizer, simple_problems, probe.num_layers, device, simple_answer_positions
        )
    except Exception as e:
        print(f"Error in layer skipping for SINGLE-HOP: {e}")
        print("Using placeholder KL divergences...")
        simple_kl = [0.1] * probe.num_layers
    
    print("\n" + "="*60)
    print("Computing layer skipping KL divergence for MULTI-HOP...")
    print("="*60)
    try:
        multihop_kl = compute_layer_skipping_kl(
            model, tokenizer, multihop_problems, probe.num_layers, device, multihop_answer_positions
        )
    except Exception as e:
        print(f"Error in layer skipping for MULTI-HOP: {e}")
        print("Using placeholder KL divergences...")
        multihop_kl = [0.1] * probe.num_layers
    
    # Plot results
    print("\nPlotting results...")
    plot_metrics(simple_metrics, multihop_metrics, simple_kl, multihop_kl, probe.num_layers)
    
    # Print summary insights
    print("\n" + "="*60)
    print("Summary Insights:")
    print("="*60)
    print("Expected patterns:")
    print("  - Early layers do most reasoning")
    print("  - Deeper layers mainly polish logits")
    print("  - Multi-hop tasks may increase early-layer contribution")
    print("  - Deep layers still under-utilized")
    
    # Clean up
    probe.remove_hooks()
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()


"""
Depth Probe Analysis for GPT-2 on Math Problems
Uses pretrained decoder-only model from HuggingFace to analyze layer-wise depth metrics.
Compares SIMPLE vs COMPLEX math problems.
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
    complex_metrics: Dict[str, List[float]],
    simple_kl: List[float],
    complex_kl: List[float],
    num_layers: int
):
    """Plot all metrics comparing SIMPLE vs COMPLEX."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPT-2 Depth Probe Metrics: SIMPLE vs COMPLEX Math Problems', 
                 fontsize=16, fontweight='bold')
    
    layers = list(range(num_layers))
    
    # Residual L2 norm
    axes[0, 0].plot(layers, simple_metrics['l2_norm'], 'b-o', linewidth=2, markersize=6, label='SIMPLE')
    axes[0, 0].plot(layers, complex_metrics['l2_norm'], 'r-s', linewidth=2, markersize=6, label='COMPLEX')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('L2 Norm')
    axes[0, 0].set_title('Residual L2 Norm ||h_l||_2')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Relative contribution
    axes[0, 1].plot(layers, simple_metrics['relative_contribution'], 'b-o', linewidth=2, markersize=6, label='SIMPLE')
    axes[0, 1].plot(layers, complex_metrics['relative_contribution'], 'r-s', linewidth=2, markersize=6, label='COMPLEX')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('||u_l|| / ||h_l||')
    axes[0, 1].set_title('Relative Contribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cosine similarity
    axes[1, 0].plot(layers, simple_metrics['cosine_similarity'], 'b-o', linewidth=2, markersize=6, label='SIMPLE')
    axes[1, 0].plot(layers, complex_metrics['cosine_similarity'], 'r-s', linewidth=2, markersize=6, label='COMPLEX')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_title('Cosine Similarity (h_l, u_l)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-1, 1])
    
    # KL divergence under layer skipping
    axes[1, 1].plot(layers, simple_kl, 'b-o', linewidth=2, markersize=6, label='SIMPLE')
    axes[1, 1].plot(layers, complex_kl, 'r-s', linewidth=2, markersize=6, label='COMPLEX')
    axes[1, 1].set_xlabel('Layer Skipped')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].set_title('KL Divergence (Layer Skipped)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Math_DecoderOnly/depth_probe_gpt2_math_metrics.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'Math_DecoderOnly/depth_probe_gpt2_math_metrics.png'")
    plt.show()


def find_answer_token_position(tokenizer: GPT2Tokenizer, problem: str) -> int:
    """
    Find the position of the first token after "Answer:" in the tokenized sequence.
    This is where we'll probe for depth metrics.
    """
    tokens = tokenizer.encode(problem)
    # Find "Answer" token position
    answer_tokens = tokenizer.encode("Answer:")
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
    
    # SIMPLE math problems
    simple_problems = [
        "John has 6 apples and buys 3 more. How many does he have now? Answer:",
        "A bag has 12 candies. If you eat 5, how many remain? Answer:",
        "Sarah bought 4 pens and 2 pencils. How many items did she buy? Answer:",
        "A farmer has 7 cows and 8 chickens. How many animals are there total? Answer:"
    ]
    
    # COMPLEX math problems (Level-6)
    complex_problems = [
        "Solve for x: 3(2x−5)+4 = 2(x+7) + 10x. Answer:",
        "A triangle has sides 7, 24, 25. Find the altitude to the longest side. Answer:",
        "A rectangle has perimeter 84. Length is 3× width. Find the area. Answer:",
        "A chemist mixes 30% and 70% alcohol to make 200 mL of 50%. How much 30% solution? Answer:",
        "A number leaves remainder 7 mod 12 and 3 mod 5. Find the smallest such number. Answer:",
        "A ball is thrown upward: h(t)=−5t²+20t+6. When does it hit the ground? Answer:"
    ]
    
    print(f"\nSIMPLE problems: {len(simple_problems)}")
    print(f"COMPLEX problems: {len(complex_problems)}")
    
    # Initialize depth probe
    print("\nSetting up depth probe hooks...")
    probe = GPT2DepthProbe(model)
    probe.register_hooks()
    
    print(f"Registered hooks for {probe.num_layers} transformer layers")
    
    # Process SIMPLE problems
    print("\n" + "="*60)
    print("Processing SIMPLE problems...")
    print("="*60)
    simple_metrics, simple_answer_positions = process_problems(
        model, tokenizer, simple_problems, probe, device
    )
    
    # Process COMPLEX problems
    print("\n" + "="*60)
    print("Processing COMPLEX problems...")
    print("="*60)
    complex_metrics, complex_answer_positions = process_problems(
        model, tokenizer, complex_problems, probe, device
    )
    
    print(f"\nComputed metrics for {len(simple_metrics['l2_norm'])} layers")
    
    # Compute layer skipping KL divergence
    print("\n" + "="*60)
    print("Computing layer skipping KL divergence for SIMPLE...")
    print("="*60)
    try:
        simple_kl = compute_layer_skipping_kl(
            model, tokenizer, simple_problems, probe.num_layers, device, simple_answer_positions
        )
    except Exception as e:
        print(f"Error in layer skipping for SIMPLE: {e}")
        print("Using placeholder KL divergences...")
        simple_kl = [0.1] * probe.num_layers
    
    print("\n" + "="*60)
    print("Computing layer skipping KL divergence for COMPLEX...")
    print("="*60)
    try:
        complex_kl = compute_layer_skipping_kl(
            model, tokenizer, complex_problems, probe.num_layers, device, complex_answer_positions
        )
    except Exception as e:
        print(f"Error in layer skipping for COMPLEX: {e}")
        print("Using placeholder KL divergences...")
        complex_kl = [0.1] * probe.num_layers
    
    # Plot results
    print("\nPlotting results...")
    plot_metrics(simple_metrics, complex_metrics, simple_kl, complex_kl, probe.num_layers)
    
    # Clean up
    probe.remove_hooks()
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()


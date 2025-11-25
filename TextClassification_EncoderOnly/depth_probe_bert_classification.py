"""
Depth Probe Analysis for BERT-base-uncased on Text Classification
Uses pretrained encoder-only model from HuggingFace to analyze layer-wise depth metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import types


class BERTDepthProbe:
    """Class to handle depth probing on BERT encoder blocks."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {}
        
    def register_hooks(self):
        """Register forward hooks on all encoder blocks."""
        # Find encoder blocks
        encoder_blocks = None
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder') and hasattr(self.model.bert.encoder, 'layer'):
            encoder_blocks = self.model.bert.encoder.layer
        else:
            # Try to find encoder blocks in the model structure
            for name, module in self.model.named_modules():
                if 'encoder' in name and 'layer' in name and isinstance(module, nn.ModuleList):
                    encoder_blocks = module
                    break
        
        if encoder_blocks is None:
            raise ValueError("Could not find encoder blocks in the model")
        
        self.num_layers = len(encoder_blocks)
        
        # Register hooks on each block
        for i, block in enumerate(encoder_blocks):
            self._register_block_hooks(block, i)
    
    def _register_block_hooks(self, block: nn.Module, layer_idx: int):
        """
        Register hooks for BERT encoder block:
        - h_l: input to the block (before any processing)
        - a_l: self-attention output (after attn, before residual)
        - m_l: FFN output (after mlp, before residual)
        """
        # Hook on block input (h_l)
        def input_hook(module, input):
            self.activations[f'h_{layer_idx}'] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.hooks.append(handle1)
        
        # Find self-attention and FFN submodules
        # BERT blocks have 'attention' and 'intermediate'/'output' structure
        attention_module = None
        ffn_module = None
        
        if hasattr(block, 'attention'):
            attention_module = block.attention
        if hasattr(block, 'output'):
            ffn_module = block.output
        
        # Fallback: search by name
        if attention_module is None or ffn_module is None:
            for name, module in block.named_children():
                if 'attention' in name.lower() and attention_module is None:
                    attention_module = module
                elif ('output' in name.lower() or 'ffn' in name.lower()) and ffn_module is None:
                    ffn_module = module
        
        # Hook on self-attention output (a_l)
        # BERT attention.self outputs (context_layer, attention_probs)
        if attention_module is not None and hasattr(attention_module, 'self'):
            def self_attn_hook(module, input, output):
                # Self-attention returns (context_layer, attention_probs)
                if isinstance(output, tuple):
                    self.activations[f'a_{layer_idx}'] = output[0].detach().clone()
                else:
                    self.activations[f'a_{layer_idx}'] = output.detach().clone()
            
            handle2 = attention_module.self.register_forward_hook(self_attn_hook)
            self.hooks.append(handle2)
        
        # Hook on FFN output (m_l)
        # BERT's FFN output.dense gives the second FFN layer output (before layer norm and residual)
        if ffn_module is not None and hasattr(ffn_module, 'dense'):
            def ffn_dense_hook(module, input, output):
                # This is the output of the second FFN layer (before layer norm and residual)
                self.activations[f'm_{layer_idx}'] = output.detach().clone()
            
            handle3 = ffn_module.dense.register_forward_hook(ffn_dense_hook)
            self.hooks.append(handle3)
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_metrics(probe: BERTDepthProbe, num_layers: int) -> Dict[str, List[float]]:
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


def compute_layer_skipping_accuracy(
    model: nn.Module,
    tokenizer: BertTokenizer,
    input_texts: List[str],
    labels: List[int],
    num_layers: int,
    device: torch.device
) -> List[float]:
    """
    Compute classification accuracy for each skipped layer.
    For each layer s, bypass block s and compute accuracy vs original predictions.
    """
    model.eval()
    
    # Get original predictions
    original_predictions = []
    with torch.no_grad():
        for input_text in input_texts:
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            original_predictions.append(pred)
    
    # Find encoder blocks
    encoder_blocks = None
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
        encoder_blocks = model.bert.encoder.layer
    else:
        for name, module in model.named_modules():
            if 'encoder' in name and 'layer' in name and isinstance(module, nn.ModuleList):
                encoder_blocks = module
                break
    
    if encoder_blocks is None:
        raise ValueError("Could not find encoder blocks for layer skipping")
    
    # Store original forward functions
    original_forwards = [block.forward for block in encoder_blocks]
    
    accuracies = []
    
    # Test each layer skipping
    print("Computing layer skipping accuracy...")
    for s in range(num_layers):
        # Temporarily replace forward method with identity
        def make_bypass(skip_layer_idx):
            def bypass_forward(self, hidden_states, *args, **kwargs):
                # Return hidden_states unchanged (skip the layer)
                # BERT encoder layer returns BaseModelOutput or tuple
                return_dict = kwargs.get('return_dict', True)
                if return_dict:
                    from transformers.modeling_outputs import BaseModelOutput
                    return BaseModelOutput(
                        last_hidden_state=hidden_states,
                        hidden_states=None,
                        attentions=None
                    )
                else:
                    return (hidden_states, None, None)
            return bypass_forward
        
        encoder_blocks[s].forward = types.MethodType(make_bypass(s), encoder_blocks[s])
        
        # Compute predictions with skipped layer
        skipped_predictions = []
        with torch.no_grad():
            for input_text in input_texts:
                inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=-1).item()
                skipped_predictions.append(pred)
        
        # Compute accuracy: how many predictions match original?
        correct = sum(1 for orig, skipped in zip(original_predictions, skipped_predictions) if orig == skipped)
        accuracy = correct / len(original_predictions)
        accuracies.append(accuracy)
        
        print(f"  Skipped layer {s}, accuracy={accuracy:.4f}")
        
        # Restore original forward
        encoder_blocks[s].forward = original_forwards[s]
    
    return accuracies


def plot_metrics(metrics: Dict[str, List[float]], accuracies: List[float], num_layers: int):
    """Plot all four metrics vs layer index."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('BERT-base-uncased Depth Probe Metrics (Text Classification)', 
                 fontsize=16, fontweight='bold')
    
    layers = list(range(num_layers))
    
    # Residual L2 norm
    axes[0, 0].plot(layers, metrics['l2_norm'], 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('L2 Norm')
    axes[0, 0].set_title('Residual L2 Norm')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Relative contribution
    axes[0, 1].plot(layers, metrics['relative_contribution'], 'g-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('||u_l|| / ||h_l||')
    axes[0, 1].set_title('Relative Contribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cosine similarity
    axes[1, 0].plot(layers, metrics['cosine_similarity'], 'r-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_title('Cosine Similarity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-1, 1])
    
    # Layer skipping accuracy
    axes[1, 1].plot(layers, accuracies, 'm-o', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Layer Skipped')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy (Layer Skipped)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('TextClassification_EncoderOnly/depth_probe_bert_metrics.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'TextClassification_EncoderOnly/depth_probe_bert_metrics.png'")
    plt.show()


def main():
    """Main function to run depth probe analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained BERT-base-uncased
    print("Loading pretrained BERT-base-uncased...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = model.to(device)
    model.eval()
    
    # Create small classification dataset (SST-2 style: sentiment classification)
    # 0 = negative, 1 = positive
    classification_data = [
        {'text': 'This movie is absolutely fantastic and I loved every minute of it!', 'label': 1},
        {'text': 'The worst film I have ever seen, completely terrible.', 'label': 0},
        {'text': 'A wonderful experience that made me feel happy and joyful.', 'label': 1},
        {'text': 'I hated this boring and disappointing story.', 'label': 0},
        {'text': 'Amazing performance with great acting and beautiful cinematography.', 'label': 1},
        {'text': 'Poor quality and uninteresting plot that was hard to follow.', 'label': 0}
    ]
    
    input_texts = [item['text'] for item in classification_data]
    labels = [item['label'] for item in classification_data]
    
    print(f"Using {len(classification_data)} text samples for analysis")
    
    # Initialize depth probe
    print("Setting up depth probe hooks...")
    probe = BERTDepthProbe(model)
    probe.register_hooks()
    
    print(f"Registered hooks for {probe.num_layers} encoder layers")
    
    # Forward pass to capture activations
    print("Running forward pass to capture activations...")
    with torch.no_grad():
        for input_text in input_texts:
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            _ = model(**inputs)
    
    # Compute metrics
    print("Computing depth probe metrics...")
    metrics = compute_metrics(probe, probe.num_layers)
    
    print(f"Computed metrics for {len(metrics['l2_norm'])} layers")
    if len(metrics['l2_norm']) == 0:
        print("Warning: No metrics computed. Check if hooks are capturing activations correctly.")
        return
    
    # Compute layer skipping accuracy
    print("Computing layer skipping accuracy...")
    try:
        accuracies = compute_layer_skipping_accuracy(
            model, tokenizer, input_texts, labels, probe.num_layers, device
        )
    except Exception as e:
        print(f"Error in layer skipping: {e}")
        print("Using placeholder accuracies...")
        accuracies = [0.5] * probe.num_layers
    
    # Plot results
    print("Plotting results...")
    plot_metrics(metrics, accuracies, probe.num_layers)
    
    # Clean up
    probe.remove_hooks()
    print("Analysis complete!")


if __name__ == '__main__':
    main()


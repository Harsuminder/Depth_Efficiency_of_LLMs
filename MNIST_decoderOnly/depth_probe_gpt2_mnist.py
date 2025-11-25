"""
Depth Probe Analysis for GPT-2 small on MNIST
Uses pretrained decoder-only model from HuggingFace to analyze layer-wise depth metrics.
MNIST images are converted to text-like sequences for GPT-2 processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


class DepthProbe:
    """Class to handle depth probing on GPT-2 transformer blocks."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {}
        
    def register_hooks(self):
        """Register forward hooks on all transformer blocks."""
        # GPT-2 has blocks in model.transformer.h
        if not hasattr(self.model, "transformer") or not hasattr(self.model.transformer, "h"):
            raise ValueError("Could not find transformer blocks in GPT-2 model")
        
        blocks = self.model.transformer.h
        self.num_layers = len(blocks)
        
        # Register hooks on each block
        for i, block in enumerate(blocks):
            self._register_block_hooks(block, i)
    
    def _register_block_hooks(self, block: nn.Module, layer_idx: int):
        """Register hooks to capture h_l, a_l, m_l for a single GPT-2 block.
        
        In GPT-2 blocks:
        - h_l: input to the block (before ln_1)
        - a_l: attention output (after attn, before residual addition)
        - m_l: MLP output (after mlp, before residual addition)
        """
        
        # Hook on block input (h_l) - capture before any processing
        def input_hook(module, input):
            self.activations[f"h_{layer_idx}"] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.hooks.append(handle1)
        
        # GPT-2 blocks have 'attn' and 'mlp' attributes
        attn_module = block.attn
        mlp_module = block.mlp
        
        # Hook on attention output (a_l)
        # In GPT-2, attention output is after the attention module but before residual
        def attn_hook(module, input, output):
            # GPT-2 attention returns tuple (output, present), we want output
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output
            self.activations[f"a_{layer_idx}"] = attn_output.detach().clone()
        
        handle2 = attn_module.register_forward_hook(attn_hook)
        self.hooks.append(handle2)
        
        # Hook on MLP output (m_l)
        def mlp_hook(module, input, output):
            self.activations[f"m_{layer_idx}"] = output.detach().clone()
        
        handle3 = mlp_module.register_forward_hook(mlp_hook)
        self.hooks.append(handle3)
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def mnist_to_text_sequence(image: torch.Tensor, num_bins: int = 16) -> str:
    """
    Convert MNIST image to text-like sequence for GPT-2.
    
    Steps:
    1. Flatten 28x28 → 784 pixel values
    2. Bucket pixel values into bins (0-15)
    3. Convert bins to string like "p3 p7 p14 p2 ..."
    """
    # Flatten image: [1, 28, 28] -> [784]
    flattened = image.view(-1)
    
    # Normalize to [0, 1] if not already
    if flattened.max() > 1.0:
        flattened = flattened / 255.0
    
    # Bucket into bins: [0, 1] -> [0, num_bins-1]
    bins = (flattened * (num_bins - 1)).long().clamp(0, num_bins - 1)
    
    # Convert to string: "p0 p5 p12 ..."
    sequence = " ".join([f"p{int(b.item())}" for b in bins])
    
    return sequence


def preprocess_mnist_batch(
    images: torch.Tensor,
    tokenizer: GPT2Tokenizer,
    max_length: int = 1024,
    num_bins: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Convert batch of MNIST images to tokenized sequences.
    
    Returns:
        Dictionary with 'input_ids' and 'attention_mask'
    """
    sequences = []
    for img in images:
        seq = mnist_to_text_sequence(img, num_bins)
        sequences.append(seq)
    
    # Tokenize all sequences
    tokenized = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    return tokenized


def compute_metrics(probe: DepthProbe, num_layers: int) -> Dict[str, List[float]]:
    """
    Compute depth probe metrics:
    1. Residual L2 norm: ||h_l||_2
    2. Relative contribution: ||u_l|| / ||h_l|| where u_l = a_l + m_l
    3. Cosine similarity: (h_l • u_l) / (||h_l|| * ||u_l||)
    """
    metrics = {
        "l2_norm": [],
        "relative_contribution": [],
        "cosine_similarity": [],
    }
    
    for l in range(num_layers):
        h_l = probe.activations.get(f"h_{l}")
        a_l = probe.activations.get(f"a_{l}")
        m_l = probe.activations.get(f"m_{l}")
        
        if h_l is None:
            continue
        
        # (1) Residual L2 norm
        # GPT-2 activations: [batch, seq_len, hidden_dim]
        l2_norm = torch.norm(h_l, p=2, dim=-1).mean().item()
        metrics["l2_norm"].append(l2_norm)
        
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
            metrics["relative_contribution"].append(rel_contrib)
            
            # (3) Cosine similarity
            dot_product = (h_l * u_l).sum(dim=-1)  # [batch, seq_len]
            cosine_per_element = dot_product / ((h_norm * u_norm) + 1e-8)
            cosine = cosine_per_element.mean().item()
            metrics["cosine_similarity"].append(cosine)
        else:
            metrics["relative_contribution"].append(np.nan)
            metrics["cosine_similarity"].append(np.nan)
    
    return metrics


def compute_layer_skipping_accuracy(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_layers: int,
) -> List[float]:
    """
    Compute accuracy when each transformer block is skipped.
    Accuracy is measured as the fraction of samples where the top predicted token
    matches the original model's top prediction.
    """
    model.eval()
    blocks = model.transformer.h

    # 1. Get original predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        original_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
        original_preds = original_logits.argmax(dim=-1)  # [batch]

    accuracies = []

    print(f"Computing layer skipping accuracy for {num_layers} layers...")

    for s in range(num_layers):
        original_block = blocks[s]

        # Skip Block: identity function
        class SkipBlock(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, hidden_states, *args, **kwargs):
                # match HF output structure: return (hidden_states, present)
                present = None
                return hidden_states, present

        # patch the block
        blocks[s] = SkipBlock()

        # run forward with layer skipped
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            skipped_logits = out.logits[:, -1, :]  # [batch, vocab_size]
            skipped_preds = skipped_logits.argmax(dim=-1)  # [batch]

        # Compute accuracy: fraction of matching predictions
        matches = (skipped_preds == original_preds).float()
        accuracy = matches.mean().item()
        accuracies.append(accuracy)

        print(f"  skipped layer {s}, accuracy={accuracy:.4f}")

        # restore original block
        blocks[s] = original_block

    print(f"Layer skipping complete: {len(accuracies)} accuracy values")
    return accuracies



def plot_metrics(metrics: Dict[str, List[float]], accuracies: List[float], num_layers: int):
    """Plot all four metrics vs layer index."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Depth Probe Metrics: GPT-2 small on MNIST", fontsize=14)
    
    # Use actual length of metrics to handle cases where some layers might be missing
    actual_num_layers = len(metrics["l2_norm"])
    layer_indices = list(range(actual_num_layers))
    
    # (1) Residual L2 norm
    if metrics["l2_norm"]:
        axes[0, 0].plot(layer_indices, metrics["l2_norm"], "o-", color="blue", markersize=6, linewidth=2)
    axes[0, 0].set_xlabel("Layer Index")
    axes[0, 0].set_ylabel("Residual L2 Norm ||h_l||_2")
    axes[0, 0].set_title("(1) Residual L2 Norm")
    axes[0, 0].grid(True, alpha=0.3)
    
    # (2) Relative contribution (filter out NaN values)
    rel_contrib = [x for x in metrics["relative_contribution"] if not np.isnan(x)]
    rel_indices = [i for i, x in enumerate(metrics["relative_contribution"]) if not np.isnan(x)]
    if rel_contrib:
        axes[0, 1].plot(rel_indices, rel_contrib, "o-", color="green", markersize=6, linewidth=2)
    axes[0, 1].set_xlabel("Layer Index")
    axes[0, 1].set_ylabel("Relative Contribution ||u_l|| / ||h_l||")
    axes[0, 1].set_title("(2) Relative Contribution")
    axes[0, 1].grid(True, alpha=0.3)
    
    # (3) Cosine similarity (filter out NaN values)
    cosine_sim = [x for x in metrics["cosine_similarity"] if not np.isnan(x)]
    cosine_indices = [i for i, x in enumerate(metrics["cosine_similarity"]) if not np.isnan(x)]
    if cosine_sim:
        axes[1, 0].plot(cosine_indices, cosine_sim, "o-", color="red", markersize=6, linewidth=2)
    axes[1, 0].set_xlabel("Layer Index")
    axes[1, 0].set_ylabel("Cosine Similarity")
    axes[1, 0].set_title("(3) Cosine Similarity (h_l • u_l) / (||h_l|| * ||u_l||)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # (4) Layer skipping (Accuracy)
    if accuracies and len(accuracies) > 0:
        acc_indices = list(range(len(accuracies)))
        axes[1, 1].plot(acc_indices, accuracies, "o-", color="purple", markersize=6, linewidth=2)
        print(f"Layer skipping: plotted {len(accuracies)} accuracy points")
    else:
        print("Warning: No layer skipping data to plot")
    axes[1, 1].set_xlabel("Layer Index (skipped)")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("(4) Layer Skipping: Prediction Accuracy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig("depth_probe_gpt2_metrics.png", dpi=150, bbox_inches="tight")
    print("Plot saved as 'depth_probe_gpt2_metrics.png'")
    plt.show()


def main():
    """Main function to run depth probe analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained GPT-2 small (do NOT train)
    print("Loading pretrained GPT-2 small...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = model.to(device)
    model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    mnist_dataset = datasets.MNIST(
        root="./data",
        train=False,  # Use test set
        download=True,
        transform=transform,
    )
    
    # Create a small batch for speed
    batch_size = 8  # Smaller batch for GPT-2
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(dataloader))
    
    # Convert MNIST images to tokenized sequences
    print("Converting MNIST images to text sequences...")
    tokenized = preprocess_mnist_batch(images, tokenizer, max_length=1024, num_bins=16)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample sequence length: {input_ids.shape[1]} tokens")
    
    # Initialize depth probe
    print("Setting up depth probe hooks...")
    probe = DepthProbe(model)
    probe.register_hooks()
    
    # Forward pass to capture activations
    print("Running forward pass to capture activations...")
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Compute metrics
    print("Computing depth probe metrics...")
    num_layers = probe.num_layers
    metrics = compute_metrics(probe, num_layers)
    
    print(f"Computed metrics for {len(metrics['l2_norm'])} layers")
    if len(metrics["l2_norm"]) == 0:
        print("Warning: No metrics computed. Check if hooks are capturing activations correctly.")
        return
    
    # Compute layer skipping accuracy
    print("Computing layer skipping accuracy...")
    try:
        accuracies = compute_layer_skipping_accuracy(model, input_ids, attention_mask, num_layers)
    except Exception as e:
        print(f"Error computing layer skipping accuracy: {e}")
        print("Continuing without layer skipping metrics...")
        import traceback
        traceback.print_exc()
        accuracies = []
    
    # Plot results
    print("Plotting results...")
    try:
        plot_metrics(metrics, accuracies, num_layers)
    except Exception as e:
        print(f"Error plotting metrics: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    probe.remove_hooks()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()

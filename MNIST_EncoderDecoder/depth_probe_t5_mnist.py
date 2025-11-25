"""
Depth Probe Analysis for T5-small on MNIST
Uses pretrained encoder-decoder model from HuggingFace to analyze layer-wise depth metrics.
MNIST images are converted to text-like sequences for T5 processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


class DepthProbe:
    """Class to handle depth probing on T5 encoder and decoder blocks."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.encoder_activations = {}
        self.decoder_activations = {}
        
    def register_hooks(self):
        """Register forward hooks on all encoder and decoder blocks."""
        # T5 has encoder and decoder blocks
        if not hasattr(self.model, "encoder") or not hasattr(self.model, "decoder"):
            raise ValueError("Could not find encoder/decoder in T5 model")
        
        encoder_blocks = self.model.encoder.block
        decoder_blocks = self.model.decoder.block
        
        self.num_encoder_layers = len(encoder_blocks)
        self.num_decoder_layers = len(decoder_blocks)
        
        # Register hooks on encoder blocks
        for i, block in enumerate(encoder_blocks):
            self._register_encoder_block_hooks(block, i)
        
        # Register hooks on decoder blocks
        for i, block in enumerate(decoder_blocks):
            self._register_decoder_block_hooks(block, i)
    
    def _register_encoder_block_hooks(self, block: nn.Module, layer_idx: int):
        """Register hooks to capture h_l, a_l, m_l for a single T5 encoder block.
        
        T5 encoder blocks have:
        - layer[0]: self-attention
        - layer[1]: feed-forward (DenseReluDense)
        """
        # Hook on block input (h_l)
        def input_hook(module, input):
            self.encoder_activations[f"h_{layer_idx}"] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.hooks.append(handle1)
        
        # T5 encoder block structure: layer[0] = self-attention, layer[1] = feed-forward
        if len(block.layer) >= 2:
            attn_module = block.layer[0]
            ff_module = block.layer[1]
            
            # Hook on attention output (a_l)
            def attn_hook(module, input, output):
                # T5 attention returns tuple (hidden_states, ...), we want hidden_states
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output
                self.encoder_activations[f"a_{layer_idx}"] = attn_output.detach().clone()
            
            handle2 = attn_module.register_forward_hook(attn_hook)
            self.hooks.append(handle2)
            
            # Hook on feed-forward output (m_l)
            def ff_hook(module, input, output):
                self.encoder_activations[f"m_{layer_idx}"] = output.detach().clone()
            
            handle3 = ff_module.register_forward_hook(ff_hook)
            self.hooks.append(handle3)
    
    def _register_decoder_block_hooks(self, block: nn.Module, layer_idx: int):
        """Register hooks to capture h_l, a_l, m_l for a single T5 decoder block.
        
        T5 decoder blocks have:
        - layer[0]: self-attention
        - layer[1]: cross-attention
        - layer[2]: feed-forward (DenseReluDense)
        
        For depth metrics:
        - a_l: output after cross-attention (which includes self-attention via residual)
        - m_l: output from feed-forward
        """
        # Hook on block input (h_l)
        def input_hook(module, input):
            self.decoder_activations[f"h_{layer_idx}"] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.hooks.append(handle1)
        
        # T5 decoder block structure
        if len(block.layer) >= 3:
            cross_attn_module = block.layer[1]  # Cross-attention comes after self-attention
            ff_module = block.layer[2]
            
            # Hook on cross-attention output (a_l)
            # This captures the combined effect of self-attention + cross-attention
            # because self-attention output is added as residual before cross-attention
            def cross_attn_hook(module, input, output):
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output
                self.decoder_activations[f"a_{layer_idx}"] = attn_output.detach().clone()
            
            handle2 = cross_attn_module.register_forward_hook(cross_attn_hook)
            self.hooks.append(handle2)
            
            # Hook on feed-forward output (m_l)
            def ff_hook(module, input, output):
                self.decoder_activations[f"m_{layer_idx}"] = output.detach().clone()
            
            handle3 = ff_module.register_forward_hook(ff_hook)
            self.hooks.append(handle3)
    
    def clear_activations(self):
        """Clear stored activations."""
        self.encoder_activations = {}
        self.decoder_activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def mnist_to_text_sequence(image: torch.Tensor, num_bins: int = 16) -> str:
    """
    Convert MNIST image to text-like sequence for T5.
    
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
    labels: torch.Tensor,
    tokenizer: T5Tokenizer,
    max_length: int = 1024,
    num_bins: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Convert batch of MNIST images and labels to tokenized sequences.
    
    Returns:
        Dictionary with encoder inputs and decoder inputs/labels
    """
    # Convert images to sequences
    encoder_sequences = []
    for img in images:
        seq = mnist_to_text_sequence(img, num_bins)
        encoder_sequences.append(seq)
    
    # Convert labels to strings (e.g., 5 -> "5")
    decoder_sequences = [str(int(label.item())) for label in labels]
    
    # Tokenize encoder inputs
    encoder_tokenized = tokenizer(
        encoder_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Tokenize decoder inputs (for T5, decoder input is the same as labels but shifted)
    decoder_tokenized = tokenizer(
        decoder_sequences,
        padding=True,
        truncation=True,
        max_length=10,
        return_tensors="pt",
    )
    
    return {
        "input_ids": encoder_tokenized["input_ids"],
        "attention_mask": encoder_tokenized["attention_mask"],
        "decoder_input_ids": decoder_tokenized["input_ids"],
        "decoder_attention_mask": decoder_tokenized["attention_mask"],
        "labels": decoder_tokenized["input_ids"],  # For T5, labels are the same as decoder_input_ids
    }


def compute_metrics(
    activations: Dict[str, torch.Tensor], num_layers: int, stack_name: str = ""
) -> Dict[str, List[float]]:
    """
    Compute depth probe metrics for encoder or decoder.
    
    Metrics:
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
        h_l = activations.get(f"h_{l}")
        a_l = activations.get(f"a_{l}")
        m_l = activations.get(f"m_{l}")
        
        if h_l is None:
            continue
        
        # (1) Residual L2 norm
        # T5 activations: [batch, seq_len, hidden_dim]
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
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    labels: torch.Tensor,
    num_layers: int,
    stack_type: str = "encoder",
) -> List[float]:
    """
    Compute accuracy when each transformer block is skipped.
    Accuracy is measured as the fraction of samples where the top predicted token
    matches the original model's top prediction.
    
    Args:
        stack_type: "encoder" or "decoder"
    """
    model.eval()
    
    if stack_type == "encoder":
        blocks = model.encoder.block
    else:
        blocks = model.decoder.block
    
    # Get original predictions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        original_logits = outputs.logits  # [batch, seq_len, vocab_size]
        # Get predictions for the last token (or first non-pad token)
        original_preds = original_logits.argmax(dim=-1)  # [batch, seq_len]
        # Compare with labels (shifted)
        original_matches = (original_preds == labels).float()
        original_accuracy = original_matches.mean().item()
    
    accuracies = []
    
    print(f"Computing layer skipping accuracy for {stack_type} ({num_layers} layers)...")
    print(f"  Original model accuracy: {original_accuracy:.4f}")
    
    for s in range(num_layers):
        original_block = blocks[s]
        original_forward = original_block.forward
        
        # Create a skip forward function that returns identity with proper tuple structure
        is_decoder_flag = (stack_type == "decoder")
        
        def skip_forward(hidden_states, *args, **kwargs):
            # T5 encoder blocks return: (hidden_states, position_bias)
            # T5 decoder blocks return: (hidden_states, past_key_value, encoder_decoder_position_bias)
            # For skipping, return hidden_states unchanged with None placeholders
            if is_decoder_flag:
                # Decoder: (hidden_states, past_key_value, encoder_decoder_position_bias)
                return (hidden_states, None, None)
            else:
                # Encoder: (hidden_states, position_bias)
                return (hidden_states, None)
        
        # Patch the forward method instead of replacing the block
        blocks[s].forward = skip_forward
        
        # Run forward with layer skipped
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            skipped_logits = outputs.logits
            skipped_preds = skipped_logits.argmax(dim=-1)
        
        # Compute accuracy: fraction of matching predictions
        matches = (skipped_preds == labels).float()
        accuracy = matches.mean().item()
        accuracies.append(accuracy)
        
        print(f"  skipped {stack_type} layer {s}, accuracy={accuracy:.4f}")
        
        # Restore original forward method
        blocks[s].forward = original_forward
    
    print(f"Layer skipping complete: {len(accuracies)} accuracy values")
    return accuracies


def plot_metrics(
    encoder_metrics: Dict[str, List[float]],
    decoder_metrics: Dict[str, List[float]],
    encoder_accuracies: List[float],
    decoder_accuracies: List[float],
    num_encoder_layers: int,
    num_decoder_layers: int,
):
    """Plot all four metrics vs layer index for encoder and decoder separately."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Depth Probe Metrics: T5-small on MNIST", fontsize=16)
    
    # Encoder plots (left column)
    encoder_layer_indices = list(range(len(encoder_metrics["l2_norm"])))
    
    # Encoder: (1) Residual L2 norm
    if encoder_metrics["l2_norm"]:
        axes[0, 0].plot(
            encoder_layer_indices,
            encoder_metrics["l2_norm"],
            "o-",
            color="blue",
            markersize=6,
            linewidth=2,
        )
    axes[0, 0].set_xlabel("Layer Index")
    axes[0, 0].set_ylabel("Residual L2 Norm ||h_l||_2")
    axes[0, 0].set_title("Encoder: (1) Residual L2 Norm")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Encoder: (2) Relative contribution
    rel_contrib = [
        x for x in encoder_metrics["relative_contribution"] if not np.isnan(x)
    ]
    rel_indices = [
        i
        for i, x in enumerate(encoder_metrics["relative_contribution"])
        if not np.isnan(x)
    ]
    if rel_contrib:
        axes[1, 0].plot(
            rel_indices, rel_contrib, "o-", color="green", markersize=6, linewidth=2
        )
    axes[1, 0].set_xlabel("Layer Index")
    axes[1, 0].set_ylabel("Relative Contribution ||u_l|| / ||h_l||")
    axes[1, 0].set_title("Encoder: (2) Relative Contribution")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Encoder: (3) Cosine similarity
    cosine_sim = [
        x for x in encoder_metrics["cosine_similarity"] if not np.isnan(x)
    ]
    cosine_indices = [
        i
        for i, x in enumerate(encoder_metrics["cosine_similarity"])
        if not np.isnan(x)
    ]
    if cosine_sim:
        axes[2, 0].plot(
            cosine_indices, cosine_sim, "o-", color="red", markersize=6, linewidth=2
        )
    axes[2, 0].set_xlabel("Layer Index")
    axes[2, 0].set_ylabel("Cosine Similarity")
    axes[2, 0].set_title("Encoder: (3) Cosine Similarity")
    axes[2, 0].grid(True, alpha=0.3)
    
    # Encoder: (4) Layer skipping (Accuracy)
    if encoder_accuracies and len(encoder_accuracies) > 0:
        acc_indices = list(range(len(encoder_accuracies)))
        axes[3, 0].plot(
            acc_indices, encoder_accuracies, "o-", color="purple", markersize=6, linewidth=2
        )
        print(f"Encoder layer skipping: plotted {len(encoder_accuracies)} points")
    else:
        print("Warning: No encoder layer skipping data to plot")
    axes[3, 0].set_xlabel("Layer Index (skipped)")
    axes[3, 0].set_ylabel("Accuracy")
    axes[3, 0].set_title("Encoder: (4) Layer Skipping: Accuracy")
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].set_ylim([0, 1.1])
    
    # Decoder plots (right column)
    decoder_layer_indices = list(range(len(decoder_metrics["l2_norm"])))
    
    # Decoder: (1) Residual L2 norm
    if decoder_metrics["l2_norm"]:
        axes[0, 1].plot(
            decoder_layer_indices,
            decoder_metrics["l2_norm"],
            "o-",
            color="blue",
            markersize=6,
            linewidth=2,
        )
    axes[0, 1].set_xlabel("Layer Index")
    axes[0, 1].set_ylabel("Residual L2 Norm ||h_l||_2")
    axes[0, 1].set_title("Decoder: (1) Residual L2 Norm")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Decoder: (2) Relative contribution
    rel_contrib = [
        x for x in decoder_metrics["relative_contribution"] if not np.isnan(x)
    ]
    rel_indices = [
        i
        for i, x in enumerate(decoder_metrics["relative_contribution"])
        if not np.isnan(x)
    ]
    if rel_contrib:
        axes[1, 1].plot(
            rel_indices, rel_contrib, "o-", color="green", markersize=6, linewidth=2
        )
    axes[1, 1].set_xlabel("Layer Index")
    axes[1, 1].set_ylabel("Relative Contribution ||u_l|| / ||h_l||")
    axes[1, 1].set_title("Decoder: (2) Relative Contribution")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Decoder: (3) Cosine similarity
    cosine_sim = [
        x for x in decoder_metrics["cosine_similarity"] if not np.isnan(x)
    ]
    cosine_indices = [
        i
        for i, x in enumerate(decoder_metrics["cosine_similarity"])
        if not np.isnan(x)
    ]
    if cosine_sim:
        axes[2, 1].plot(
            cosine_indices, cosine_sim, "o-", color="red", markersize=6, linewidth=2
        )
    axes[2, 1].set_xlabel("Layer Index")
    axes[2, 1].set_ylabel("Cosine Similarity")
    axes[2, 1].set_title("Decoder: (3) Cosine Similarity")
    axes[2, 1].grid(True, alpha=0.3)
    
    # Decoder: (4) Layer skipping (Accuracy)
    if decoder_accuracies and len(decoder_accuracies) > 0:
        acc_indices = list(range(len(decoder_accuracies)))
        axes[3, 1].plot(
            acc_indices, decoder_accuracies, "o-", color="purple", markersize=6, linewidth=2
        )
        print(f"Decoder layer skipping: plotted {len(decoder_accuracies)} points")
    else:
        print("Warning: No decoder layer skipping data to plot")
    axes[3, 1].set_xlabel("Layer Index (skipped)")
    axes[3, 1].set_ylabel("Accuracy")
    axes[3, 1].set_title("Decoder: (4) Layer Skipping: Accuracy")
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig("depth_probe_t5_metrics.png", dpi=150, bbox_inches="tight")
    print("Plot saved as 'depth_probe_t5_metrics.png'")
    plt.show()


def main():
    """Main function to run depth probe analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained T5-small (do NOT train)
    print("Loading pretrained T5-small...")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = model.to(device)
    model.eval()
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    mnist_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    # Create a small batch for speed
    batch_size = 8
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(dataloader))
    
    # Convert MNIST images and labels to tokenized sequences
    print("Converting MNIST images to text sequences...")
    tokenized = preprocess_mnist_batch(images, labels, tokenizer, max_length=1024, num_bins=16)
    
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    decoder_input_ids = tokenized["decoder_input_ids"].to(device)
    decoder_attention_mask = tokenized["decoder_attention_mask"].to(device)
    labels = tokenized["labels"].to(device)
    
    print(f"Encoder input shape: {input_ids.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}")
    
    # Initialize depth probe
    print("Setting up depth probe hooks...")
    probe = DepthProbe(model)
    probe.register_hooks()
    
    # Forward pass to capture activations
    print("Running forward pass to capture activations...")
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
    
    # Compute metrics for encoder
    print("Computing encoder depth probe metrics...")
    encoder_metrics = compute_metrics(
        probe.encoder_activations, probe.num_encoder_layers, "encoder"
    )
    print(f"Computed encoder metrics for {len(encoder_metrics['l2_norm'])} layers")
    
    # Compute metrics for decoder
    print("Computing decoder depth probe metrics...")
    decoder_metrics = compute_metrics(
        probe.decoder_activations, probe.num_decoder_layers, "decoder"
    )
    print(f"Computed decoder metrics for {len(decoder_metrics['l2_norm'])} layers")
    
    if len(encoder_metrics["l2_norm"]) == 0 or len(decoder_metrics["l2_norm"]) == 0:
        print("Warning: No metrics computed. Check if hooks are capturing activations correctly.")
        return
    
    # Compute layer skipping accuracy for encoder
    print("Computing encoder layer skipping accuracy...")
    try:
        encoder_accuracies = compute_layer_skipping_accuracy(
            model,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
            probe.num_encoder_layers,
            "encoder",
        )
    except Exception as e:
        print(f"Error computing encoder layer skipping: {e}")
        import traceback
        traceback.print_exc()
        encoder_accuracies = []
    
    # Compute layer skipping accuracy for decoder
    print("Computing decoder layer skipping accuracy...")
    try:
        decoder_accuracies = compute_layer_skipping_accuracy(
            model,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
            probe.num_decoder_layers,
            "decoder",
        )
    except Exception as e:
        print(f"Error computing decoder layer skipping: {e}")
        import traceback
        traceback.print_exc()
        decoder_accuracies = []
    
    # Plot results
    print("Plotting results...")
    try:
        plot_metrics(
            encoder_metrics,
            decoder_metrics,
            encoder_accuracies,
            decoder_accuracies,
            probe.num_encoder_layers,
            probe.num_decoder_layers,
        )
    except Exception as e:
        print(f"Error plotting metrics: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    probe.remove_hooks()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()


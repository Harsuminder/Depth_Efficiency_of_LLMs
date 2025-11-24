"""
Depth Probe Analysis for ViT-tiny on MNIST
Uses pretrained encoder-only model from timm to analyze layer-wise depth metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import types


class DepthProbe:
    """Class to handle depth probing on ViT encoder blocks."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {}
        
    def register_hooks(self):
        """Register forward hooks on all encoder blocks."""
        # Find all transformer blocks in the model
        blocks = None
        if hasattr(self.model, 'blocks'):
            blocks = self.model.blocks
        else:
            # Try to find blocks in the model structure
            for name, module in self.model.named_modules():
                if 'blocks' in name and isinstance(module, nn.ModuleList):
                    blocks = module
                    break
        
        if blocks is None:
            raise ValueError("Could not find encoder blocks in the model")
        
        self.num_layers = len(blocks)
        
        # Register hooks on each block
        for i, block in enumerate(blocks):
            self._register_block_hooks(block, i)
    
    def _register_block_hooks(self, block: nn.Module, layer_idx: int):
        """Register hooks to capture h_l, a_l, m_l for a single block.
        
        In timm ViT blocks:
        - h_l: input to the block (before norm1)
        - a_l: attention output (after attn, before residual addition)
        - m_l: MLP output (after mlp, before residual addition)
        """
        
        # Hook on block input (h_l) - capture before any processing
        # Pre-hooks only receive (module, input), not output
        def input_hook(module, input):
            self.activations[f'h_{layer_idx}'] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.hooks.append(handle1)
        
        # Find attention and MLP submodules
        # timm ViT blocks typically have 'attn' and 'mlp' attributes
        attn_module = None
        mlp_module = None
        
        # Try direct attribute access first (most common in timm)
        if hasattr(block, 'attn'):
            attn_module = block.attn
        if hasattr(block, 'mlp'):
            mlp_module = block.mlp
        
        # Fallback: search by name
        if attn_module is None or mlp_module is None:
            for name, module in block.named_children():
                if 'attn' in name.lower() and attn_module is None:
                    attn_module = module
                elif ('mlp' in name.lower() or 'fc' in name.lower()) and 'attn' not in name.lower() and mlp_module is None:
                    mlp_module = module
        
        # Hook on attention output (a_l)
        if attn_module is not None:
            def attn_hook(module, input, output):
                # Output from attention module (before residual)
                self.activations[f'a_{layer_idx}'] = output.detach().clone()
            
            handle2 = attn_module.register_forward_hook(attn_hook)
            self.hooks.append(handle2)
        else:
            print(f"Warning: Could not find attention module in layer {layer_idx}")
        
        # Hook on MLP output (m_l)
        if mlp_module is not None:
            def mlp_hook(module, input, output):
                # Output from MLP module (before residual)
                self.activations[f'm_{layer_idx}'] = output.detach().clone()
            
            handle3 = mlp_module.register_forward_hook(mlp_hook)
            self.hooks.append(handle3)
        else:
            print(f"Warning: Could not find MLP module in layer {layer_idx}")
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def preprocess_mnist(image: torch.Tensor) -> torch.Tensor:
    """
    Preprocess MNIST image:
    - Resize to 224x224
    - Repeat channels to 3
    """
    # Resize to 224x224
    image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    # Repeat channels to 3
    image = image.repeat(1, 3, 1, 1)
    return image.squeeze(0)


def compute_metrics(probe: DepthProbe, num_layers: int) -> Dict[str, List[float]]:
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
        
        # (1) Residual L2 norm
        l2_norm = torch.norm(h_l, p=2, dim=-1).mean().item()
        metrics['l2_norm'].append(l2_norm)
        
        if a_l is not None and m_l is not None:
            # Ensure shapes match (ViT activations: [batch, seq_len, hidden_dim])
            # Take minimum dimensions to handle any shape mismatches
            batch_size = min(h_l.shape[0], a_l.shape[0], m_l.shape[0])
            
            # Handle 2D or 3D tensors
            if len(h_l.shape) == 3:
                seq_len = min(h_l.shape[1], a_l.shape[1], m_l.shape[1])
                h_l = h_l[:batch_size, :seq_len, :]
                a_l = a_l[:batch_size, :seq_len, :]
                m_l = m_l[:batch_size, :seq_len, :]
            else:
                # 2D case: [batch, hidden]
                hidden_dim = min(h_l.shape[-1], a_l.shape[-1], m_l.shape[-1])
                h_l = h_l[:batch_size, :hidden_dim]
                a_l = a_l[:batch_size, :hidden_dim]
                m_l = m_l[:batch_size, :hidden_dim]
            
            # u_l = a_l + m_l
            u_l = a_l + m_l
            
            # (2) Relative contribution: ||u_l|| / ||h_l||
            # Compute L2 norm along last dimension (hidden dim)
            u_norm = torch.norm(u_l, p=2, dim=-1)  # Shape: [batch, seq_len] or [batch]
            h_norm = torch.norm(h_l, p=2, dim=-1)  # Shape: [batch, seq_len] or [batch]
            rel_contrib = (u_norm / (h_norm + 1e-8)).mean().item()
            metrics['relative_contribution'].append(rel_contrib)
            
            # (3) Cosine similarity: (h_l • u_l) / (||h_l|| * ||u_l||)
            # Compute dot product along last dimension
            dot_product = (h_l * u_l).sum(dim=-1)  # Shape: [batch, seq_len] or [batch]
            cosine_per_element = dot_product / ((h_norm * u_norm) + 1e-8)
            cosine = cosine_per_element.mean().item()
            metrics['cosine_similarity'].append(cosine)
        else:
            # If a_l or m_l not captured, append NaN
            metrics['relative_contribution'].append(np.nan)
            metrics['cosine_similarity'].append(np.nan)
    
    return metrics


def compute_layer_skipping(model: nn.Module, images: torch.Tensor, num_layers: int) -> List[float]:
    """
    Compute KL divergence for each skipped layer.
    For each layer s, temporarily bypass block s (h_{s+1} = h_s) and compute KL divergence.
    """
    model.eval()
    
    # Get original logits
    with torch.no_grad():
        original_logits = model(images)
        original_probs = F.softmax(original_logits, dim=-1)
    
    kl_divs = []
    
    # Find blocks
    blocks = None
    if hasattr(model, 'blocks'):
        blocks = model.blocks
    else:
        for name, module in model.named_modules():
            if 'blocks' in name and isinstance(module, nn.ModuleList):
                blocks = module
                break
    
    if blocks is None:
        raise ValueError("Could not find encoder blocks for layer skipping")
    
    # Store original forward functions
    original_forwards = []
    for block in blocks:
        original_forwards.append(block.forward)
    
    # Create bypass forward function
    def make_bypass_forward(original_forward):
        def bypass_forward(self, x):
            return x
        return bypass_forward
    
    for s in range(num_layers):
        # Temporarily replace forward method with identity function
        blocks[s].forward = types.MethodType(lambda self, x: x, blocks[s])
        
        # Compute logits with skipped layer
        with torch.no_grad():
            skipped_logits = model(images)
        
        # Compute KL divergence: KL(skipped || original)
        kl_div = F.kl_div(
            F.log_softmax(skipped_logits, dim=-1),
            original_probs,
            reduction='batchmean'
        ).item()
        kl_divs.append(kl_div)
        
        # Restore original forward
        blocks[s].forward = original_forwards[s]
    
    return kl_divs


def plot_metrics(metrics: Dict[str, List[float]], kl_divs: List[float], num_layers: int):
    """Plot all four metrics vs layer index."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Depth Probe Metrics: ViT-tiny on MNIST', fontsize=14)
    
    # Use actual length of metrics to handle cases where some layers might be missing
    actual_num_layers = len(metrics['l2_norm'])
    layer_indices = list(range(actual_num_layers))
    
    # (1) Residual L2 norm
    if metrics['l2_norm']:
        axes[0, 0].plot(layer_indices, metrics['l2_norm'], 'o-', color='blue')
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Residual L2 Norm ||h_l||_2')
    axes[0, 0].set_title('(1) Residual L2 Norm')
    axes[0, 0].grid(True, alpha=0.3)
    
    # (2) Relative contribution (filter out NaN values)
    rel_contrib = [x for x in metrics['relative_contribution'] if not np.isnan(x)]
    rel_indices = [i for i, x in enumerate(metrics['relative_contribution']) if not np.isnan(x)]
    if rel_contrib:
        axes[0, 1].plot(rel_indices, rel_contrib, 'o-', color='green')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Relative Contribution ||u_l|| / ||h_l||')
    axes[0, 1].set_title('(2) Relative Contribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # (3) Cosine similarity (filter out NaN values)
    cosine_sim = [x for x in metrics['cosine_similarity'] if not np.isnan(x)]
    cosine_indices = [i for i, x in enumerate(metrics['cosine_similarity']) if not np.isnan(x)]
    if cosine_sim:
        axes[1, 0].plot(cosine_indices, cosine_sim, 'o-', color='red')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_title('(3) Cosine Similarity (h_l • u_l) / (||h_l|| * ||u_l||)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # (4) Layer skipping (KL divergence)
    if kl_divs:
        kl_indices = list(range(len(kl_divs)))
        axes[1, 1].plot(kl_indices, kl_divs, 'o-', color='purple')
    axes[1, 1].set_xlabel('Layer Index (skipped)')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].set_title('(4) Layer Skipping: KL Divergence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('depth_probe_metrics.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'depth_probe_metrics.png'")
    plt.show()


def main():
    """Main function to run depth probe analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained ViT-tiny (do NOT train)
    print("Loading pretrained ViT-tiny...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    mnist_dataset = datasets.MNIST(
        root='./data',
        train=False,  # Use test set
        download=True,
        transform=transform
    )
    
    # Create a small batch
    batch_size = 32
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(dataloader))
    
    # Preprocess MNIST images
    print("Preprocessing MNIST images...")
    processed_images = []
    for img in images:
        processed_img = preprocess_mnist(img)
        processed_images.append(processed_img)
    processed_images = torch.stack(processed_images).to(device)
    
    # Initialize depth probe
    print("Setting up depth probe hooks...")
    probe = DepthProbe(model)
    probe.register_hooks()
    
    # Forward pass to capture activations
    print("Running forward pass to capture activations...")
    with torch.no_grad():
        _ = model(processed_images)
    
    # Compute metrics
    print("Computing depth probe metrics...")
    num_layers = probe.num_layers
    metrics = compute_metrics(probe, num_layers)
    
    print(f"Computed metrics for {len(metrics['l2_norm'])} layers")
    if len(metrics['l2_norm']) == 0:
        print("Warning: No metrics computed. Check if hooks are capturing activations correctly.")
        return
    
    # Compute layer skipping KL divergence
    print("Computing layer skipping KL divergence...")
    try:
        kl_divs = compute_layer_skipping(model, processed_images, num_layers)
    except Exception as e:
        print(f"Error computing layer skipping: {e}")
        print("Continuing without layer skipping metrics...")
        kl_divs = []
    
    # Plot results
    print("Plotting results...")
    try:
        plot_metrics(metrics, kl_divs, num_layers)
    except Exception as e:
        print(f"Error plotting metrics: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    probe.remove_hooks()
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()


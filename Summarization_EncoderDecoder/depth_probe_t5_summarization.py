"""
Depth Probe Analysis for T5-small on Text Summarization
Uses pretrained encoder-decoder model from HuggingFace to analyze layer-wise depth metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import types
from rouge_score import rouge_scorer


class T5DepthProbe:
    """Class to handle depth probing on T5 encoder and decoder blocks."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.encoder_hooks = []
        self.decoder_hooks = []
        self.encoder_activations = {}
        self.decoder_activations = {}
        
    def register_hooks(self):
        """Register forward hooks on all encoder and decoder blocks."""
        # Find encoder blocks
        encoder_blocks = None
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'block'):
            encoder_blocks = self.model.encoder.block
        else:
            for name, module in self.model.named_modules():
                if 'encoder' in name and 'block' in name and isinstance(module, nn.ModuleList):
                    encoder_blocks = module
                    break
        
        if encoder_blocks is None:
            raise ValueError("Could not find encoder blocks in the model")
        
        self.num_encoder_layers = len(encoder_blocks)
        
        # Find decoder blocks
        decoder_blocks = None
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'block'):
            decoder_blocks = self.model.decoder.block
        else:
            for name, module in self.model.named_modules():
                if 'decoder' in name and 'block' in name and isinstance(module, nn.ModuleList):
                    decoder_blocks = module
                    break
        
        if decoder_blocks is None:
            raise ValueError("Could not find decoder blocks in the model")
        
        self.num_decoder_layers = len(decoder_blocks)
        
        # Register hooks on encoder blocks
        for i, block in enumerate(encoder_blocks):
            self._register_encoder_block_hooks(block, i)
        
        # Register hooks on decoder blocks
        for i, block in enumerate(decoder_blocks):
            self._register_decoder_block_hooks(block, i)
    
    def _register_encoder_block_hooks(self, block: nn.Module, layer_idx: int):
        """
        Register hooks for encoder block:
        - h_l: input to the block (before norm1)
        - a_l: self-attention output (after attn, before residual)
        - m_l: FFN output (after mlp, before residual)
        """
        # Hook on block input (h_l)
        def input_hook(module, input):
            self.encoder_activations[f'h_{layer_idx}'] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.encoder_hooks.append(handle1)
        
        # Find self-attention and FFN submodules
        # T5 blocks have 'layer' which contains [0] self-attn, [1] cross-attn (only in decoder), [2] FFN
        if hasattr(block, 'layer'):
            layers = block.layer
            # Encoder: layer[0] = self-attn, layer[1] = FFN
            if len(layers) >= 2:
                self_attn_module = layers[0]
                ffn_module = layers[1]
                
                # Hook on self-attention output (a_l)
                def attn_hook(module, input, output):
                    # T5 attention returns tuple: (output, attention_weights)
                    if isinstance(output, tuple):
                        self.encoder_activations[f'a_{layer_idx}'] = output[0].detach().clone()
                    else:
                        self.encoder_activations[f'a_{layer_idx}'] = output.detach().clone()
                
                handle2 = self_attn_module.register_forward_hook(attn_hook)
                self.encoder_hooks.append(handle2)
                
                # Hook on FFN output (m_l)
                def ffn_hook(module, input, output):
                    self.encoder_activations[f'm_{layer_idx}'] = output.detach().clone()
                
                handle3 = ffn_module.register_forward_hook(ffn_hook)
                self.encoder_hooks.append(handle3)
    
    def _register_decoder_block_hooks(self, block: nn.Module, layer_idx: int):
        """
        Register hooks for decoder block:
        - h_l: input to the block (before norm1)
        - a_l: self-attention output (after self-attn, before residual)
        - ca_l: cross-attention output (after cross-attn, before residual)
        - m_l: FFN output (after mlp, before residual)
        """
        # Hook on block input (h_l)
        def input_hook(module, input):
            self.decoder_activations[f'h_{layer_idx}'] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.decoder_hooks.append(handle1)
        
        # Find self-attention, cross-attention, and FFN submodules
        if hasattr(block, 'layer'):
            layers = block.layer
            # Decoder: layer[0] = self-attn, layer[1] = cross-attn, layer[2] = FFN
            if len(layers) >= 3:
                self_attn_module = layers[0]
                cross_attn_module = layers[1]
                ffn_module = layers[2]
                
                # Hook on self-attention output (a_l)
                def self_attn_hook(module, input, output):
                    if isinstance(output, tuple):
                        self.decoder_activations[f'a_{layer_idx}'] = output[0].detach().clone()
                    else:
                        self.decoder_activations[f'a_{layer_idx}'] = output.detach().clone()
                
                handle2 = self_attn_module.register_forward_hook(self_attn_hook)
                self.decoder_hooks.append(handle2)
                
                # Hook on cross-attention output (ca_l)
                def cross_attn_hook(module, input, output):
                    if isinstance(output, tuple):
                        self.decoder_activations[f'ca_{layer_idx}'] = output[0].detach().clone()
                    else:
                        self.decoder_activations[f'ca_{layer_idx}'] = output.detach().clone()
                
                handle3 = cross_attn_module.register_forward_hook(cross_attn_hook)
                self.decoder_hooks.append(handle3)
                
                # Hook on FFN output (m_l)
                def ffn_hook(module, input, output):
                    self.decoder_activations[f'm_{layer_idx}'] = output.detach().clone()
                
                handle4 = ffn_module.register_forward_hook(ffn_hook)
                self.decoder_hooks.append(handle4)
    
    def clear_activations(self):
        """Clear stored activations."""
        self.encoder_activations = {}
        self.decoder_activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.encoder_hooks + self.decoder_hooks:
            hook.remove()
        self.encoder_hooks = []
        self.decoder_hooks = []


def compute_encoder_metrics(probe: T5DepthProbe, num_layers: int) -> Dict[str, List[float]]:
    """
    Compute depth probe metrics for encoder:
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
        h_l = probe.encoder_activations.get(f'h_{l}')
        a_l = probe.encoder_activations.get(f'a_{l}')
        m_l = probe.encoder_activations.get(f'm_{l}')
        
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


def compute_decoder_metrics(probe: T5DepthProbe, num_layers: int) -> Dict[str, List[float]]:
    """
    Compute depth probe metrics for decoder:
    1. Residual L2 norm: ||h_l||_2
    2. Relative contribution: ||u_l|| / ||h_l|| where u_l = a_l + ca_l + m_l
    3. Cosine similarity: (h_l • u_l) / (||h_l|| * ||u_l||)
    """
    metrics = {
        'l2_norm': [],
        'relative_contribution': [],
        'cosine_similarity': []
    }
    
    for l in range(num_layers):
        h_l = probe.decoder_activations.get(f'h_{l}')
        a_l = probe.decoder_activations.get(f'a_{l}')
        ca_l = probe.decoder_activations.get(f'ca_{l}')
        m_l = probe.decoder_activations.get(f'm_{l}')
        
        if h_l is None:
            continue
        
        # (1) Residual L2 norm
        l2_norm = torch.norm(h_l, p=2, dim=-1).mean().item()
        metrics['l2_norm'].append(l2_norm)
        
        if a_l is not None and ca_l is not None and m_l is not None:
            # Ensure shapes match
            batch_size = min(h_l.shape[0], a_l.shape[0], ca_l.shape[0], m_l.shape[0])
            seq_len = min(h_l.shape[1], a_l.shape[1], ca_l.shape[1], m_l.shape[1])
            hidden_dim = min(h_l.shape[2], a_l.shape[2], ca_l.shape[2], m_l.shape[2])
            
            h_l = h_l[:batch_size, :seq_len, :hidden_dim]
            a_l = a_l[:batch_size, :seq_len, :hidden_dim]
            ca_l = ca_l[:batch_size, :seq_len, :hidden_dim]
            m_l = m_l[:batch_size, :seq_len, :hidden_dim]
            
            # u_l = a_l + ca_l + m_l
            u_l = a_l + ca_l + m_l
            
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


def compute_rouge1_fscore(prediction: str, reference: str) -> float:
    """Compute ROUGE-1 F-score between prediction and reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rouge1'].fmeasure


def compute_layer_skipping_rouge(
    model: nn.Module,
    tokenizer: T5Tokenizer,
    input_texts: List[str],
    target_texts: List[str],
    num_encoder_layers: int,
    num_decoder_layers: int,
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """
    Compute ROUGE-1 F-score for each skipped layer.
    Returns: (encoder_rouge_scores, decoder_rouge_scores)
    """
    model.eval()
    
    # Get original summaries
    original_summaries = []
    with torch.no_grad():
        for input_text in input_texts:
            # Add "summarize: " prefix for T5
            input_text_prefixed = f"summarize: {input_text}"
            input_ids = tokenizer.encode(input_text_prefixed, return_tensors='pt', max_length=512, truncation=True).to(device)
            outputs = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            original_summaries.append(summary)
    
    # Find encoder and decoder blocks
    encoder_blocks = None
    decoder_blocks = None
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        encoder_blocks = model.encoder.block
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
        decoder_blocks = model.decoder.block
    
    if encoder_blocks is None or decoder_blocks is None:
        raise ValueError("Could not find encoder or decoder blocks")
    
    # Store original forward functions
    encoder_original_forwards = [block.forward for block in encoder_blocks]
    decoder_original_forwards = [block.forward for block in decoder_blocks]
    
    encoder_rouge_scores = []
    decoder_rouge_scores = []
    
    # Test encoder layer skipping
    print("Computing encoder layer skipping ROUGE scores...")
    for s in range(num_encoder_layers):
        # Temporarily replace forward method with identity
        # Encoder block forward signature: forward(hidden_states, attention_mask=None, ...)
        # Returns: (hidden_states, self_attn_weights, ...)
        def make_bypass_enc(skip_layer_idx):
            def bypass_encoder_forward(self, hidden_states, *args, **kwargs):
                # Return hidden_states unchanged (skip the layer)
                # T5 encoder blocks return a tuple, typically (hidden_states, attention_weights, ...)
                return (hidden_states, None)
            return bypass_encoder_forward
        
        encoder_blocks[s].forward = types.MethodType(make_bypass_enc(s), encoder_blocks[s])
        
        # Generate summaries with skipped layer
        skipped_summaries = []
        with torch.no_grad():
            for input_text in input_texts:
                # Add "summarize: " prefix for T5
                input_text_prefixed = f"summarize: {input_text}"
                input_ids = tokenizer.encode(input_text_prefixed, return_tensors='pt', max_length=512, truncation=True).to(device)
                outputs = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                skipped_summaries.append(summary)
        
        # Compute average ROUGE-1 F-score
        rouge_scores = []
        for orig, skipped, target in zip(original_summaries, skipped_summaries, target_texts):
            # Compare skipped summary to target (reference)
            rouge = compute_rouge1_fscore(skipped, target)
            rouge_scores.append(rouge)
        
        encoder_rouge_scores.append(np.mean(rouge_scores))
        
        # Restore original forward
        encoder_blocks[s].forward = encoder_original_forwards[s]
    
    # Test decoder layer skipping
    print("Computing decoder layer skipping ROUGE scores...")
    for s in range(num_decoder_layers):
        # Temporarily replace forward method with identity
        # Decoder block forward signature: forward(hidden_states, attention_mask=None, encoder_hidden_states=None, ...)
        # Returns: (hidden_states, self_attn_weights, cross_attn_weights, ...)
        def make_bypass(skip_layer_idx):
            def bypass_decoder_forward(self, hidden_states, *args, **kwargs):
                # Return hidden_states unchanged (skip the layer)
                # T5 decoder blocks return a tuple, typically (hidden_states, self_attn_weights, cross_attn_weights, ...)
                return (hidden_states, None, None)
            return bypass_decoder_forward
        
        decoder_blocks[s].forward = types.MethodType(make_bypass(s), decoder_blocks[s])
        
        # Generate summaries with skipped layer
        skipped_summaries = []
        with torch.no_grad():
            for input_text in input_texts:
                # Add "summarize: " prefix for T5
                input_text_prefixed = f"summarize: {input_text}"
                input_ids = tokenizer.encode(input_text_prefixed, return_tensors='pt', max_length=512, truncation=True).to(device)
                outputs = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                skipped_summaries.append(summary)
        
        # Compute average ROUGE-1 F-score
        rouge_scores = []
        for orig, skipped, target in zip(original_summaries, skipped_summaries, target_texts):
            rouge = compute_rouge1_fscore(skipped, target)
            rouge_scores.append(rouge)
        
        decoder_rouge_scores.append(np.mean(rouge_scores))
        
        # Restore original forward
        decoder_blocks[s].forward = decoder_original_forwards[s]
    
    return encoder_rouge_scores, decoder_rouge_scores


def plot_metrics(
    encoder_metrics: Dict[str, List[float]],
    decoder_metrics: Dict[str, List[float]],
    encoder_rouge: List[float],
    decoder_rouge: List[float],
    num_encoder_layers: int,
    num_decoder_layers: int
):
    """Plot all metrics for encoder and decoder separately."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle('T5-small Depth Probe Metrics', fontsize=16, fontweight='bold')
    
    # Encoder plots
    encoder_layers = list(range(num_encoder_layers))
    
    # Encoder: Residual L2 norm
    axes[0, 0].plot(encoder_layers, encoder_metrics['l2_norm'], 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('L2 Norm')
    axes[0, 0].set_title('Encoder: Residual L2 Norm')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Encoder: Relative contribution
    axes[0, 1].plot(encoder_layers, encoder_metrics['relative_contribution'], 'g-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('||u_l|| / ||h_l||')
    axes[0, 1].set_title('Encoder: Relative Contribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Encoder: Cosine similarity
    axes[0, 2].plot(encoder_layers, encoder_metrics['cosine_similarity'], 'r-o', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Cosine Similarity')
    axes[0, 2].set_title('Encoder: Cosine Similarity')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([-1, 1])
    
    # Encoder: ROUGE-1 F-score
    axes[0, 3].plot(encoder_layers, encoder_rouge, 'm-o', linewidth=2, markersize=6)
    axes[0, 3].set_xlabel('Layer Skipped')
    axes[0, 3].set_ylabel('ROUGE-1 F-score')
    axes[0, 3].set_title('Encoder: ROUGE-1 (Layer Skipped)')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Decoder plots
    decoder_layers = list(range(num_decoder_layers))
    
    # Decoder: Residual L2 norm
    axes[1, 0].plot(decoder_layers, decoder_metrics['l2_norm'], 'b-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].set_title('Decoder: Residual L2 Norm')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Decoder: Relative contribution
    axes[1, 1].plot(decoder_layers, decoder_metrics['relative_contribution'], 'g-o', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('||u_l|| / ||h_l||')
    axes[1, 1].set_title('Decoder: Relative Contribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Decoder: Cosine similarity
    axes[1, 2].plot(decoder_layers, decoder_metrics['cosine_similarity'], 'r-o', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel('Layer')
    axes[1, 2].set_ylabel('Cosine Similarity')
    axes[1, 2].set_title('Decoder: Cosine Similarity')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([-1, 1])
    
    # Decoder: ROUGE-1 F-score
    axes[1, 3].plot(decoder_layers, decoder_rouge, 'm-o', linewidth=2, markersize=6)
    axes[1, 3].set_xlabel('Layer Skipped')
    axes[1, 3].set_ylabel('ROUGE-1 F-score')
    axes[1, 3].set_title('Decoder: ROUGE-1 (Layer Skipped)')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Summarization_EncoderDecoder/depth_probe_t5_metrics.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'Summarization_EncoderDecoder/depth_probe_t5_metrics.png'")
    plt.show()


def main():
    """Main function to run depth probe analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained T5-small
    print("Loading pretrained T5-small...")
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = model.to(device)
    model.eval()
    
    # Define sample texts and summaries
    sample_texts = [
        "Scientists have discovered a new species of deep-sea fish that glows in the dark. The fish was found at a depth of 3,000 meters in the Pacific Ocean. Researchers say it uses bioluminescence to attract prey and communicate with other fish.",
        "A major tech company announced plans to build a new data center powered entirely by renewable energy. The facility will use solar and wind power, reducing carbon emissions by 90 percent. Construction is expected to begin next year.",
        "Researchers at a leading university have developed a new AI system that can diagnose medical conditions with high accuracy. The system was trained on millions of medical records and can identify diseases faster than traditional methods.",
        "A new study shows that regular exercise can improve mental health and reduce anxiety. The research followed 1,000 participants over two years and found significant improvements in mood and cognitive function.",
        "A breakthrough in battery technology could lead to electric vehicles with much longer range. The new battery design uses advanced materials that store more energy and charge faster than current batteries."
    ]
    
    target_summaries = [
        "New glowing deep-sea fish species discovered in Pacific Ocean.",
        "Tech company to build renewable energy data center.",
        "AI system developed for faster medical diagnosis.",
        "Study links exercise to improved mental health.",
        "New battery technology could extend electric vehicle range."
    ]
    
    print(f"Using {len(sample_texts)} text samples for analysis")
    
    # Initialize depth probe
    print("Setting up depth probe hooks...")
    probe = T5DepthProbe(model)
    probe.register_hooks()
    
    print(f"Registered hooks for {probe.num_encoder_layers} encoder layers and {probe.num_decoder_layers} decoder layers")
    
    # Forward pass to capture activations
    print("Running forward pass to capture activations...")
    with torch.no_grad():
        for input_text in sample_texts:
            # Prepare input for T5 (add "summarize: " prefix)
            input_text_prefixed = f"summarize: {input_text}"
            input_ids = tokenizer.encode(input_text_prefixed, return_tensors='pt', max_length=512, truncation=True).to(device)
            
            # Run encoder to capture encoder activations
            encoder_outputs = model.encoder(input_ids=input_ids)
            
            # For decoder, create decoder input IDs (start with pad token for generation)
            decoder_start_token_id = model.config.decoder_start_token_id
            decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device=device) * decoder_start_token_id
            
            # Run decoder to capture decoder activations
            _ = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=None
            )
    
    # Compute encoder metrics
    print("Computing encoder depth probe metrics...")
    encoder_metrics = compute_encoder_metrics(probe, probe.num_encoder_layers)
    
    # Compute decoder metrics
    print("Computing decoder depth probe metrics...")
    decoder_metrics = compute_decoder_metrics(probe, probe.num_decoder_layers)
    
    print(f"Computed metrics for {len(encoder_metrics['l2_norm'])} encoder layers and {len(decoder_metrics['l2_norm'])} decoder layers")
    
    # Compute layer skipping ROUGE scores
    print("Computing layer skipping ROUGE scores (this may take a while)...")
    encoder_rouge, decoder_rouge = compute_layer_skipping_rouge(
        model, tokenizer, sample_texts, target_summaries,
        probe.num_encoder_layers, probe.num_decoder_layers, device
    )
    
    # Plot results
    print("Plotting results...")
    plot_metrics(encoder_metrics, decoder_metrics, encoder_rouge, decoder_rouge,
                 probe.num_encoder_layers, probe.num_decoder_layers)
    
    # Clean up
    probe.remove_hooks()
    print("Analysis complete!")


if __name__ == '__main__':
    main()


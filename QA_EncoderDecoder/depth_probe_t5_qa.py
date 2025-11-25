"""
Depth Probe Analysis for T5-small on Question Answering (QA)
Compares simple factoid vs compositional QA tasks using depth metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import types
from collections import Counter
import re


class T5QADepthProbe:
    """Class to handle depth probing on T5 encoder and decoder blocks for QA."""
    
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
        """Register hooks for encoder block."""
        # Hook on block input (h_l)
        def input_hook(module, input):
            self.encoder_activations[f'h_{layer_idx}'] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.encoder_hooks.append(handle1)
        
        # Find self-attention and FFN submodules
        if hasattr(block, 'layer'):
            layers = block.layer
            if len(layers) >= 2:
                self_attn_module = layers[0]
                ffn_module = layers[1]
                
                # Hook on self-attention output (a_l)
                def attn_hook(module, input, output):
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
        """Register hooks for decoder block."""
        # Hook on block input (h_l)
        def input_hook(module, input):
            self.decoder_activations[f'h_{layer_idx}'] = input[0].detach().clone()
        
        handle1 = block.register_forward_pre_hook(input_hook)
        self.decoder_hooks.append(handle1)
        
        # Find self-attention, cross-attention, and FFN submodules
        if hasattr(block, 'layer'):
            layers = block.layer
            if len(layers) >= 3:
                self_attn_module = layers[0]
                cross_attn_module = layers[1]
                ffn_module = layers[2]
                
                # Hook on self-attention output (sa_l)
                def self_attn_hook(module, input, output):
                    if isinstance(output, tuple):
                        self.decoder_activations[f'sa_{layer_idx}'] = output[0].detach().clone()
                    else:
                        self.decoder_activations[f'sa_{layer_idx}'] = output.detach().clone()
                
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


def compute_encoder_metrics(probe: T5QADepthProbe, num_layers: int) -> Dict[str, List[float]]:
    """Compute depth probe metrics for encoder."""
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
        
        # (1) Residual L2 norm
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
            u_norm = torch.norm(u_l, p=2, dim=-1)
            h_norm = torch.norm(h_l, p=2, dim=-1)
            rel_contrib = (u_norm / (h_norm + 1e-8)).mean().item()
            metrics['relative_contribution'].append(rel_contrib)
            
            # (3) Cosine similarity
            dot_product = (h_l * u_l).sum(dim=-1)
            cosine_per_element = dot_product / ((h_norm * torch.norm(u_l, p=2, dim=-1)) + 1e-8)
            cosine = cosine_per_element.mean().item()
            metrics['cosine_similarity'].append(cosine)
        else:
            metrics['relative_contribution'].append(np.nan)
            metrics['cosine_similarity'].append(np.nan)
    
    return metrics


def compute_decoder_metrics(probe: T5QADepthProbe, num_layers: int) -> Dict[str, List[float]]:
    """Compute depth probe metrics for decoder."""
    metrics = {
        'l2_norm': [],
        'relative_contribution': [],
        'cosine_similarity': [],
        'cross_attn_contribution': []
    }
    
    for l in range(num_layers):
        h_l = probe.decoder_activations.get(f'h_{l}')
        sa_l = probe.decoder_activations.get(f'sa_{l}')
        ca_l = probe.decoder_activations.get(f'ca_{l}')
        m_l = probe.decoder_activations.get(f'm_{l}')
        
        if h_l is None:
            continue
        
        # (1) Residual L2 norm
        l2_norm = torch.norm(h_l, p=2, dim=-1).mean().item()
        metrics['l2_norm'].append(l2_norm)
        
        if sa_l is not None and ca_l is not None and m_l is not None:
            # Ensure shapes match
            batch_size = min(h_l.shape[0], sa_l.shape[0], ca_l.shape[0], m_l.shape[0])
            seq_len = min(h_l.shape[1], sa_l.shape[1], ca_l.shape[1], m_l.shape[1])
            hidden_dim = min(h_l.shape[2], sa_l.shape[2], ca_l.shape[2], m_l.shape[2])
            
            h_l = h_l[:batch_size, :seq_len, :hidden_dim]
            sa_l = sa_l[:batch_size, :seq_len, :hidden_dim]
            ca_l = ca_l[:batch_size, :seq_len, :hidden_dim]
            m_l = m_l[:batch_size, :seq_len, :hidden_dim]
            
            # u_l = sa_l + ca_l + m_l
            u_l = sa_l + ca_l + m_l
            
            # (2) Relative contribution: ||u_l|| / ||h_l||
            u_norm = torch.norm(u_l, p=2, dim=-1)
            h_norm = torch.norm(h_l, p=2, dim=-1)
            rel_contrib = (u_norm / (h_norm + 1e-8)).mean().item()
            metrics['relative_contribution'].append(rel_contrib)
            
            # (3) Cosine similarity
            dot_product = (h_l * u_l).sum(dim=-1)
            cosine_per_element = dot_product / ((h_norm * torch.norm(u_l, p=2, dim=-1)) + 1e-8)
            cosine = cosine_per_element.mean().item()
            metrics['cosine_similarity'].append(cosine)
            
            # (4) Cross-attention contribution: ||ca_l|| / ||h_l||
            ca_norm = torch.norm(ca_l, p=2, dim=-1)
            ca_contrib = (ca_norm / (h_norm + 1e-8)).mean().item()
            metrics['cross_attn_contribution'].append(ca_contrib)
        else:
            metrics['relative_contribution'].append(np.nan)
            metrics['cosine_similarity'].append(np.nan)
            metrics['cross_attn_contribution'].append(np.nan)
    
    return metrics


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set('.,!?;:')
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0.0
    recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_layer_skipping_f1(
    model: nn.Module,
    tokenizer: T5Tokenizer,
    input_texts: List[str],
    target_texts: List[str],
    num_encoder_layers: int,
    num_decoder_layers: int,
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """
    Compute F1 score for each skipped layer.
    Returns: (encoder_f1_scores, decoder_f1_scores)
    """
    model.eval()
    
    # Get original answers
    original_answers = []
    with torch.no_grad():
        for input_text in input_texts:
            input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
            outputs = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            original_answers.append(answer)
    
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
    
    encoder_f1_scores = []
    decoder_f1_scores = []
    
    # Test encoder layer skipping
    print("Computing encoder layer skipping F1 scores...")
    for s in range(num_encoder_layers):
        # Temporarily replace forward method with identity
        def make_bypass_enc(skip_layer_idx):
            def bypass_encoder_forward(self, hidden_states, *args, **kwargs):
                return (hidden_states, None)
            return bypass_encoder_forward
        
        encoder_blocks[s].forward = types.MethodType(make_bypass_enc(s), encoder_blocks[s])
        
        # Generate answers with skipped layer
        skipped_answers = []
        with torch.no_grad():
            for input_text in input_texts:
                input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
                outputs = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                skipped_answers.append(answer)
        
        # Compute average F1 score
        f1_scores = []
        for skipped, target in zip(skipped_answers, target_texts):
            f1 = f1_score(skipped, target)
            f1_scores.append(f1)
        
        encoder_f1_scores.append(np.mean(f1_scores))
        
        # Restore original forward
        encoder_blocks[s].forward = encoder_original_forwards[s]
    
    # Test decoder layer skipping
    print("Computing decoder layer skipping F1 scores...")
    for s in range(num_decoder_layers):
        # Temporarily replace forward method with identity
        def make_bypass(skip_layer_idx):
            def bypass_decoder_forward(self, hidden_states, *args, **kwargs):
                return (hidden_states, None, None)
            return bypass_decoder_forward
        
        decoder_blocks[s].forward = types.MethodType(make_bypass(s), decoder_blocks[s])
        
        # Generate answers with skipped layer
        skipped_answers = []
        with torch.no_grad():
            for input_text in input_texts:
                input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
                outputs = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                skipped_answers.append(answer)
        
        # Compute average F1 score
        f1_scores = []
        for skipped, target in zip(skipped_answers, target_texts):
            f1 = f1_score(skipped, target)
            f1_scores.append(f1)
        
        decoder_f1_scores.append(np.mean(f1_scores))
        
        # Restore original forward
        decoder_blocks[s].forward = decoder_original_forwards[s]
    
    return encoder_f1_scores, decoder_f1_scores


def plot_metrics(
    simple_encoder_metrics: Dict[str, List[float]],
    comp_encoder_metrics: Dict[str, List[float]],
    simple_decoder_metrics: Dict[str, List[float]],
    comp_decoder_metrics: Dict[str, List[float]],
    simple_encoder_f1: List[float],
    comp_encoder_f1: List[float],
    simple_decoder_f1: List[float],
    comp_decoder_f1: List[float],
    num_encoder_layers: int,
    num_decoder_layers: int
):
    """Plot all metrics comparing simple vs compositional QA."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('T5-small Depth Probe Metrics: Simple Factoid vs Compositional QA', 
                 fontsize=16, fontweight='bold')
    
    encoder_layers = list(range(num_encoder_layers))
    decoder_layers = list(range(num_decoder_layers))
    
    # Encoder: Residual L2 norm
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(encoder_layers, simple_encoder_metrics['l2_norm'], 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax1.plot(encoder_layers, comp_encoder_metrics['l2_norm'], 'r--s', 
             linewidth=2, markersize=6, label='Compositional')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('L2 Norm')
    ax1.set_title('Encoder: Residual L2 Norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Encoder: Relative contribution
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(encoder_layers, simple_encoder_metrics['relative_contribution'], 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax2.plot(encoder_layers, comp_encoder_metrics['relative_contribution'], 'r--s', 
             linewidth=2, markersize=6, label='Compositional')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('||u_l|| / ||h_l||')
    ax2.set_title('Encoder: Relative Contribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Encoder: Cosine similarity
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(encoder_layers, simple_encoder_metrics['cosine_similarity'], 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax3.plot(encoder_layers, comp_encoder_metrics['cosine_similarity'], 'r--s', 
             linewidth=2, markersize=6, label='Compositional')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Encoder: Cosine Similarity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-1, 1])
    
    # Encoder: Layer skipping F1
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(encoder_layers, simple_encoder_f1, 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax4.plot(encoder_layers, comp_encoder_f1, 'r--s', 
             linewidth=2, markersize=6, label='Compositional')
    ax4.set_xlabel('Layer Skipped')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('Encoder: F1 (Layer Skipped)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Decoder: Residual L2 norm
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(decoder_layers, simple_decoder_metrics['l2_norm'], 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax5.plot(decoder_layers, comp_decoder_metrics['l2_norm'], 'r--s', 
             linewidth=2, markersize=6, label='Compositional')
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('L2 Norm')
    ax5.set_title('Decoder: Residual L2 Norm')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Decoder: Relative contribution
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(decoder_layers, simple_decoder_metrics['relative_contribution'], 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax6.plot(decoder_layers, comp_decoder_metrics['relative_contribution'], 'r--s', 
             linewidth=2, markersize=6, label='Compositional')
    ax6.set_xlabel('Layer')
    ax6.set_ylabel('||u_l|| / ||h_l||')
    ax6.set_title('Decoder: Relative Contribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Decoder: Cosine similarity
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(decoder_layers, simple_decoder_metrics['cosine_similarity'], 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax7.plot(decoder_layers, comp_decoder_metrics['cosine_similarity'], 'r--s', 
             linewidth=2, markersize=6, label='Compositional')
    ax7.set_xlabel('Layer')
    ax7.set_ylabel('Cosine Similarity')
    ax7.set_title('Decoder: Cosine Similarity')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([-1, 1])
    
    # Decoder: Layer skipping F1
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(decoder_layers, simple_decoder_f1, 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax8.plot(decoder_layers, comp_decoder_f1, 'r--s', 
             linewidth=2, markersize=6, label='Compositional')
    ax8.set_xlabel('Layer Skipped')
    ax8.set_ylabel('F1 Score')
    ax8.set_title('Decoder: F1 (Layer Skipped)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Decoder: Cross-attention contribution (Simple)
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(decoder_layers, simple_decoder_metrics['cross_attn_contribution'], 'b-o', 
             linewidth=2, markersize=6, label='Simple')
    ax9.set_xlabel('Layer')
    ax9.set_ylabel('||ca_l|| / ||h_l||')
    ax9.set_title('Decoder: Cross-Attn Contribution (Simple)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Decoder: Cross-attention contribution (Compositional)
    ax10 = plt.subplot(3, 4, 10)
    ax10.plot(decoder_layers, comp_decoder_metrics['cross_attn_contribution'], 'r--s', 
              linewidth=2, markersize=6, label='Compositional')
    ax10.set_xlabel('Layer')
    ax10.set_ylabel('||ca_l|| / ||h_l||')
    ax10.set_title('Decoder: Cross-Attn Contribution (Compositional)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # Decoder: Cross-attention contribution (Comparison)
    ax11 = plt.subplot(3, 4, 11)
    ax11.plot(decoder_layers, simple_decoder_metrics['cross_attn_contribution'], 'b-o', 
              linewidth=2, markersize=6, label='Simple')
    ax11.plot(decoder_layers, comp_decoder_metrics['cross_attn_contribution'], 'r--s', 
              linewidth=2, markersize=6, label='Compositional')
    ax11.set_xlabel('Layer')
    ax11.set_ylabel('||ca_l|| / ||h_l||')
    ax11.set_title('Decoder: Cross-Attn Contribution (Comparison)')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Summary comparison
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    summary_text = f"""
    Summary:
    - Simple QA: 2 factoid questions
    - Compositional QA: 2 multi-hop questions
    - Encoder layers: {num_encoder_layers}
    - Decoder layers: {num_decoder_layers}
    """
    ax12.text(0.1, 0.5, summary_text, fontsize=10, 
              verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig('QA_EncoderDecoder/depth_probe_t5_qa_metrics.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'QA_EncoderDecoder/depth_probe_t5_qa_metrics.png'")
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
    
    # Define QA dataset: 2 simple factoid + 2 compositional
    simple_qa = [
        {
            'input': 'question: What is the capital of France? context: Paris is the capital and largest city of France. It is located on the Seine River.',
            'target': 'Paris'
        },
        {
            'input': 'question: Who wrote Romeo and Juliet? context: William Shakespeare wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth.',
            'target': 'William Shakespeare'
        }
    ]
    
    compositional_qa = [
        {
            'input': 'question: What is the sum of the population of Tokyo and New York? context: Tokyo has a population of 14 million people. New York has a population of 8 million people.',
            'target': '22 million'
        },
        {
            'input': 'question: Which city is larger, the one with 5 million people or the one with 3 million people? context: London has a population of 5 million. Berlin has a population of 3 million.',
            'target': 'London'
        }
    ]
    
    simple_inputs = [item['input'] for item in simple_qa]
    simple_targets = [item['target'] for item in simple_qa]
    comp_inputs = [item['input'] for item in compositional_qa]
    comp_targets = [item['target'] for item in compositional_qa]
    
    print(f"Simple QA: {len(simple_qa)} questions")
    print(f"Compositional QA: {len(compositional_qa)} questions")
    
    # Initialize depth probe
    print("Setting up depth probe hooks...")
    probe = T5QADepthProbe(model)
    probe.register_hooks()
    
    print(f"Registered hooks for {probe.num_encoder_layers} encoder layers and {probe.num_decoder_layers} decoder layers")
    
    # Process simple QA
    print("\nProcessing simple factoid QA...")
    probe.clear_activations()
    with torch.no_grad():
        for input_text in simple_inputs:
            input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
            encoder_outputs = model.encoder(input_ids=input_ids)
            decoder_start_token_id = model.config.decoder_start_token_id
            decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device=device) * decoder_start_token_id
            _ = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=None
            )
    
    print("Computing simple QA encoder metrics...")
    simple_encoder_metrics = compute_encoder_metrics(probe, probe.num_encoder_layers)
    print("Computing simple QA decoder metrics...")
    simple_decoder_metrics = compute_decoder_metrics(probe, probe.num_decoder_layers)
    
    # Process compositional QA
    print("\nProcessing compositional QA...")
    probe.clear_activations()
    with torch.no_grad():
        for input_text in comp_inputs:
            input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
            encoder_outputs = model.encoder(input_ids=input_ids)
            decoder_start_token_id = model.config.decoder_start_token_id
            decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device=device) * decoder_start_token_id
            _ = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=None
            )
    
    print("Computing compositional QA encoder metrics...")
    comp_encoder_metrics = compute_encoder_metrics(probe, probe.num_encoder_layers)
    print("Computing compositional QA decoder metrics...")
    comp_decoder_metrics = compute_decoder_metrics(probe, probe.num_decoder_layers)
    
    # Compute layer skipping F1 scores
    print("\nComputing layer skipping F1 scores (this may take a while)...")
    print("Simple QA layer skipping...")
    simple_encoder_f1, simple_decoder_f1 = compute_layer_skipping_f1(
        model, tokenizer, simple_inputs, simple_targets,
        probe.num_encoder_layers, probe.num_decoder_layers, device
    )
    
    print("Compositional QA layer skipping...")
    comp_encoder_f1, comp_decoder_f1 = compute_layer_skipping_f1(
        model, tokenizer, comp_inputs, comp_targets,
        probe.num_encoder_layers, probe.num_decoder_layers, device
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_metrics(
        simple_encoder_metrics, comp_encoder_metrics,
        simple_decoder_metrics, comp_decoder_metrics,
        simple_encoder_f1, comp_encoder_f1,
        simple_decoder_f1, comp_decoder_f1,
        probe.num_encoder_layers, probe.num_decoder_layers
    )
    
    # Clean up
    probe.remove_hooks()
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()


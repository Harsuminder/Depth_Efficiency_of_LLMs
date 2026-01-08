# Depth Efficiency of Large Language Models

This project investigates the depth efficiency of transformer-based language models across different architectures and tasks. Through systematic depth probing analysis, we examine how information flows through model layers and assess the contribution of each layer to the overall model performance.

## Overview

Understanding how deep neural networks utilize their layers is crucial for model optimization and efficiency. This research project applies depth probing techniques to analyze three major transformer architectures:

- **Encoder-only models** (BERT, Vision Transformer)
- **Decoder-only models** (GPT-2)
- **Encoder-decoder models** (T5)

The analysis spans multiple tasks including image classification, text classification, question answering, summarization, natural language inference, and mathematical reasoning.

## Methodology

Depth probing involves instrumenting transformer models with forward hooks to capture intermediate activations at each layer. We compute several metrics to quantify layer-wise contributions:

1. **Residual L2 Norm**: Measures the magnitude of residual connections (||h_l||_2)
2. **Relative Contribution**: Ratio of update magnitude to residual magnitude (||u_l|| / ||h_l||)
3. **Cosine Similarity**: Alignment between residual and update vectors ((h_l • u_l) / (||h_l|| * ||u_l||))
4. **Layer Skipping**: Impact of removing individual layers on model output (measured via KL divergence, accuracy, or F1 score)

These metrics help identify which layers are most critical for task performance and reveal patterns in how different architectures utilize depth.

## Project Structure

The project is organized by architecture type and task:

```
├── MNIST_EncoderOnly/          # Vision Transformer on MNIST
├── MNIST_DecoderOnly/          # GPT-2 on MNIST (converted to text)
├── MNIST_EncoderDecoder/        # T5 on MNIST
├── TextClassification_EncoderOnly/  # BERT on text classification
├── NaturalLanguageInference_EncoderOnly/  # BERT on NLI
├── QA_EncoderDecoder/          # T5 on question answering
├── QA_Multihop_decoderOnly/    # GPT-2 on multi-hop QA
├── Summarization_EncoderDecoder/  # T5 on summarization
└── Math_DecoderOnly/            # GPT-2 on mathematical reasoning
```

Each directory contains:
- A Python script implementing depth probing for that specific model-task combination
- Generated metrics visualization (PNG files)

## Requirements

The project uses the following key dependencies:

- PyTorch
- Transformers (HuggingFace)
- torchvision
- matplotlib
- numpy
- timm (for Vision Transformer)

Install dependencies using pip:

```bash
pip install torch torchvision transformers matplotlib numpy timm
```

## Usage

Each experiment can be run independently. Navigate to the relevant directory and execute the Python script:

```bash
cd MNIST_DecoderOnly
python depth_probe_gpt2_mnist.py
```

The script will:
1. Load a pretrained model from HuggingFace
2. Prepare the dataset for the specific task
3. Register forward hooks to capture layer activations
4. Run inference to collect activation data
5. Compute depth probe metrics
6. Perform layer skipping experiments
7. Generate visualization plots

Results are saved as PNG files in the respective directories.

## Key Findings

The depth probing analysis reveals several consistent patterns across the experiments:

- Across nine experiments and three transformer architectures, depth usage is not a universal property of transformers — it depends strongly on the model architecture.
  - **Encoders** make substantial use of depth.
  - **Decoder-only** models show very limited depth utilization.
  - **Encoder–decoder** models fall in between.

- Task complexity rarely activates deeper decoder layers.
  - This holds even for tasks involving math, multi-hop reasoning, or compositional question answering.

- Most of the effective computation happens in the early layers.
  - Deeper layers primarily serve to refine and stabilize the output logits rather than perform new reasoning.

## Limitations

- Experiments use pretrained models without fine-tuning, so results reflect general-purpose representations rather than task-optimized ones
- Small sample sizes are used for computational efficiency
- Some experiments convert non-text inputs (e.g., MNIST images) to text sequences, which may affect model behavior

## Future Work

Potential extensions include:
- Fine-tuning models on specific tasks before depth analysis
- Larger-scale experiments with more samples
- Comparative analysis across model sizes
- Integration of attention pattern analysis with depth metrics


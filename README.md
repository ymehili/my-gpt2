# 🤖 PyTorch GPT-2 Implementation

<div align="center">

![GPT-2 Model](https://img.shields.io/badge/Model-GPT--2-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Last Updated](https://img.shields.io/badge/Last%20Updated-March%202025-lightgrey)

</div>

A clean, efficient implementation of the GPT-2 language model architecture in PyTorch, supporting training, fine-tuning, and text generation.

## ✨ Features

- **Complete Implementation**: Full PyTorch implementation of GPT-2 architecture
- **Model Variants**: Support for small (124M), medium (355M), large (774M), and XL (1.5B) models
- **Training**: Easily train or fine-tune models on custom datasets
- **Pre-trained Weights**: Import weights from Hugging Face's transformers library
- **Text Generation**: Generate text with various sampling strategies
- **Tensorboard Integration**: Monitor training progress with detailed logging

## 🔧 Installation

Clone the repository and install the required dependencies:

```bash
git clone
cd llm
pip install -r requirements.txt
```

### Dependencies

- torch (≥1.9.0)
- numpy (≥1.19.5)
- transformers (≥4.18.0)
- tqdm (≥4.62.3)
- datasets (≥2.0.0)
- matplotlib (≥3.4.3)
- tensorboard (≥2.8.0)

## 🚀 Usage

### Importing Pre-trained Weights

Import pre-trained weights from Hugging Face's transformers:

```bash
python import_pretrained.py --model_size small --output_path pretrained_model.pt
```

Options:
- `--model_size`: Model size (small, medium, large, xl)
- `--output_path`: Path to save the model with pre-trained weights
- `--no_cuda`: Disable CUDA

### Training

Train or fine-tune the model on your own dataset:

```bash
python train.py --data_path your_data.txt --model_size small
```

Key options:
- `--data_path`: Path to training data file (text)
- `--model_size`: Model size (small, medium, large, xl)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--save_dir`: Directory to save checkpoints
- `--pretrained`: Path to pre-trained model to continue training

### Text Generation

Generate text using a trained model:

```bash
python generate.py --model_path checkpoints/gpt2_small_best.pt --prompt "Once upon a time"
```

Key options:
- `--model_path`: Path to the trained model checkpoint
- `--model_size`: Model size (must match the saved model)
- `--prompt`: Text prompt to start generation
- `--max_length`: Maximum length of generated text
- `--temperature`: Temperature for sampling (higher = more random)
- `--top_k`: Top-k sampling parameter

## 🏗️ Architecture

The implementation follows the original GPT-2 architecture:

- Transformer-based language model with decoder-only structure
- Multi-head causal self-attention
- Layer normalization and residual connections
- Adaptive token and positional embeddings
- Support for different model sizes:
  - Small: 124M parameters (12 layers, 768-dim)
  - Medium: 355M parameters (24 layers, 1024-dim)
  - Large: 774M parameters (36 layers, 1280-dim)
  - XL: 1.5B parameters (48 layers, 1600-dim)

## 📊 Performance Tips

- Use a GPU with enough VRAM for your chosen model size
- For large models, consider gradient accumulation to simulate larger batch sizes
- Enable Flash Attention if using PyTorch 2.0+ for faster training
- Use lower precision (FP16) for larger models with appropriate hardware
- Start with a smaller model for faster iteration during development

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import GPT2, create_gpt2_small, create_gpt2_medium, create_gpt2_large, create_gpt2_xl

def load_pretrained_weights(custom_model, pretrained_model_name):
    """
    Load pretrained weights from Hugging Face's transformers into our custom model.
    
    Args:
        custom_model: Our custom GPT-2 model implementation
        pretrained_model_name: Name of the pretrained model (e.g., 'gpt2', 'gpt2-medium')
    
    Returns:
        The custom model with pretrained weights
    """
    # Load the pretrained model from Hugging Face
    pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
    
    # Get the state dict from the pretrained model
    pretrained_dict = pretrained_model.state_dict()
    
    # Get the state dict from our custom model
    custom_dict = custom_model.state_dict()
    
    # Map parameter names from Hugging Face model to our custom model
    mapping = {
        # Transformer embeddings
        'transformer.wte.weight': 'transformer.wte.weight',
        'transformer.wpe.weight': 'transformer.wpe.weight',
        
        # Layer norm weights at the end
        'transformer.ln_f.weight': 'transformer.ln_f.weight',
        'transformer.ln_f.bias': 'transformer.ln_f.bias',
        
        # LM head
        'lm_head.weight': 'lm_head.weight',
    }
    
    # Add mappings for each transformer block
    for i in range(custom_model.config.n_layer):
        # Attention weights
        mapping[f'transformer.h.{i}.ln_1.weight'] = f'transformer.h.{i}.ln_1.weight'
        mapping[f'transformer.h.{i}.ln_1.bias'] = f'transformer.h.{i}.ln_1.bias'
        mapping[f'transformer.h.{i}.attn.c_attn.weight'] = f'transformer.h.{i}.attn.c_attn.weight'
        mapping[f'transformer.h.{i}.attn.c_attn.bias'] = f'transformer.h.{i}.attn.c_attn.bias'
        mapping[f'transformer.h.{i}.attn.c_proj.weight'] = f'transformer.h.{i}.attn.c_proj.weight'
        mapping[f'transformer.h.{i}.attn.c_proj.bias'] = f'transformer.h.{i}.attn.c_proj.bias'
        
        # MLP weights
        mapping[f'transformer.h.{i}.ln_2.weight'] = f'transformer.h.{i}.ln_2.weight'
        mapping[f'transformer.h.{i}.ln_2.bias'] = f'transformer.h.{i}.ln_2.bias'
        mapping[f'transformer.h.{i}.mlp.c_fc.weight'] = f'transformer.h.{i}.mlp.c_fc.weight'
        mapping[f'transformer.h.{i}.mlp.c_fc.bias'] = f'transformer.h.{i}.mlp.c_fc.bias'
        mapping[f'transformer.h.{i}.mlp.c_proj.weight'] = f'transformer.h.{i}.mlp.c_proj.weight'
        mapping[f'transformer.h.{i}.mlp.c_proj.bias'] = f'transformer.h.{i}.mlp.c_proj.bias'
    
    # Create a new state dict with the mapped parameter names
    new_dict = {}
    for key, value in pretrained_dict.items():
        if key in mapping:
            new_key = mapping[key]
            if new_key in custom_dict and custom_dict[new_key].shape == value.shape:
                new_dict[new_key] = value
            else:
                print(f"Skipping parameter {key} due to shape mismatch or missing key")
    
    # Load the parameters into our custom model
    custom_model.load_state_dict(new_dict, strict=False)
    
    print(f"Successfully loaded pretrained weights from {pretrained_model_name}")
    
    return custom_model

def main():
    parser = argparse.ArgumentParser(description='Import pretrained GPT-2 weights from Hugging Face models')
    
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large', 'xl'],
                       help='GPT-2 model size to load')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the model with pretrained weights')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Map model size to Hugging Face model name and our model creator function
    model_mapping = {
        'small': ('gpt2', create_gpt2_small),
        'medium': ('gpt2-medium', create_gpt2_medium),
        'large': ('gpt2-large', create_gpt2_large),
        'xl': ('gpt2-xl', create_gpt2_xl)
    }
    
    hf_model_name, create_func = model_mapping[args.model_size]
    print(f"Loading {args.model_size} GPT-2 model from Hugging Face: {hf_model_name}")
    
    # Create our custom model
    custom_model = create_func().to(device)
    
    # Load pretrained weights
    custom_model = load_pretrained_weights(custom_model, hf_model_name)
    
    # Save the model
    torch.save(custom_model.state_dict(), args.output_path)
    print(f"Model saved to: {args.output_path}")
    
    # Quick test with a sample input
    tokenizer = GPT2Tokenizer.from_pretrained(hf_model_name)
    
    # Generate a short sample text
    prompt = "Once upon a time"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    custom_model.eval()
    with torch.no_grad():
        output = custom_model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
            top_k=50
        )
    
    generated_text = tokenizer.decode(output[0].tolist())
    print(f"\nSample generation with prompt '{prompt}':")
    print(generated_text)

if __name__ == "__main__":
    main()
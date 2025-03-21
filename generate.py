import argparse
import torch
from transformers import GPT2Tokenizer
from model import GPT2, create_gpt2_small, create_gpt2_medium, create_gpt2_large, create_gpt2_xl

def load_model(model, model_path, device):
    """
    Load model from checkpoint.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, device='cuda'):
    """
    Generate text using the GPT-2 model.
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    generated_text = tokenizer.decode(output[0].tolist())
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='GPT-2 Text Generator')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--model_size', type=str, default='small', 
                        choices=['small', 'medium', 'large', 'xl'],
                        help='GPT-2 model size (must match the saved model)')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Prompt for text generation')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter (0 = no filtering)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Initialize model based on size
    if args.model_size == 'small':
        model = create_gpt2_small()
    elif args.model_size == 'medium':
        model = create_gpt2_medium()
    elif args.model_size == 'large':
        model = create_gpt2_large()
    elif args.model_size == 'xl':
        model = create_gpt2_xl()
    
    model = model.to(device)
    
    # Load trained model
    try:
        model = load_model(model, args.model_path, device)
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate text
    print("\nGenerating text with the following parameters:")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max length: {args.max_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print("\nGenerated Text:")
    
    generated_text = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        max_length=args.max_length, 
        temperature=args.temperature, 
        top_k=args.top_k, 
        device=device
    )
    
    print(generated_text)
    print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
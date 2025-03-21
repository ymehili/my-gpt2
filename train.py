import os
import argparse
import time
import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer

from model import GPT2, create_gpt2_small, create_gpt2_medium, create_gpt2_large, create_gpt2_xl

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

def load_dataset(data_path, tokenizer, block_size=1024):
    """
    Load and tokenize text data.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize the text
    encoded_text = tokenizer.encode(text)
    
    # Create the dataset
    data = TextDataset(encoded_text, block_size)
    
    return data

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, args):
    """
    Training loop for GPT-2 model.
    """
    writer = SummaryWriter(args.log_dir)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(train_dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            if batch_idx % args.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
                progress_bar.write(f'| epoch {epoch+1:3d} | {batch_idx:5d}/{len(train_dataloader):5d} batches | '
                     f'lr {lr:.6f} | ms/batch {ms_per_batch:5.2f} | '
                     f'loss {loss.item():.2f}')
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_dataloader, device)
        writer.add_scalar('val/loss', val_loss, epoch)
        
        print('-' * 89)
        print(f'| end of epoch {epoch+1:3d} | time: {time.time() - start_time:5.2f}s | '
              f'valid loss {val_loss:5.2f}')
        print('-' * 89)
        
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.save_dir, f"gpt2_{args.model_size}_best.pt")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_model(model, args.save_dir, f"gpt2_{args.model_size}_epoch{epoch+1}.pt")
    
    writer.close()

def evaluate(model, dataloader, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
    
    return total_loss / len(dataloader.dataset)

def save_model(model, save_dir, filename):
    """
    Save model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, filename))
    print(f"Model saved to {os.path.join(save_dir, filename)}")

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
    parser = argparse.ArgumentParser(description='GPT-2 Training Script')
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large', 'xl'],
                        help='GPT-2 model size')
    parser.add_argument('--block_size', type=int, default=1024,
                        help='Block size for input sequences')
    
    # Training parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data file')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save tensorboard logs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Interval for logging training progress')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Interval for saving model checkpoints')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Portion of data to use for training (rest for validation)')
    
    # Generation parameters
    parser.add_argument('--generate', action='store_true',
                        help='Generate text after training')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Prompt for text generation')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load dataset
    data = load_dataset(args.data_path, tokenizer, args.block_size)
    
    # Split into train and validation
    train_size = int(args.train_split * len(data))
    val_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
    
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    if args.model_size == 'small':
        model = create_gpt2_small()
    elif args.model_size == 'medium':
        model = create_gpt2_medium()
    elif args.model_size == 'large':
        model = create_gpt2_large()
    elif args.model_size == 'xl':
        model = create_gpt2_xl()
    
    model = model.to(device)
    
    # Load pretrained weights if specified
    if args.pretrained:
        model = load_model(model, args.pretrained, device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / args.warmup_steps)
    )
    
    # Train the model
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, args)
    
    # Generate text if requested
    if args.generate:
        generated_text = generate_text(
            model, 
            tokenizer, 
            args.prompt, 
            max_length=args.max_length, 
            temperature=args.temperature, 
            top_k=args.top_k, 
            device=device
        )
        print("\nGenerated Text:")
        print(generated_text)

if __name__ == "__main__":
    main()
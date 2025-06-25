import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
from collections import Counter
from torch.utils.data import Dataset, DataLoader


# Dataset for Text Sequences

class TextDataset(Dataset):
    """Custom dataset for text sequences with padding"""
    def __init__(self, sentences, word2idx, max_len=25):
        self.sentences = sentences
        self.word2idx = word2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # Convert words to indices with special tokens
        encoded = [self.word2idx["<sos>"]] + [self.word2idx.get(w, self.word2idx["<unk>"]) for w in sentence.split()] + [self.word2idx["<eos>"]]
        
        # Pad or truncate to fixed length
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        else:
            encoded = encoded + [self.word2idx["<pad>"]] * (self.max_len - len(encoded))
        
        return torch.tensor(encoded, dtype=torch.long)

def load_dataset(data_paths, max_sentences=3000):
    """Load JSON datasets and extract headlines"""
    sentences = []
    
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    for data_path in data_paths:
        print(f"Loading: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'headline' in data:
                        headline = data['headline'].lower().strip()
                        if headline and len(headline.split()) <= 25:
                            sentences.append(headline)
                    if len(sentences) >= max_sentences:
                        break
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(sentences)} headlines")
    
    return sentences

def build_vocabulary(sentences, min_freq=2, max_vocab_size=3000):
    """Build word-to-index mapping from sentences"""
    word_counts = Counter()
    for sentence in sentences:
        words = sentence.lower().split()
        word_counts.update(words)
    
    # Initialize with special tokens
    word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
    
    # Add frequent words
    for word, count in word_counts.most_common(max_vocab_size):
        if count >= min_freq and len(word2idx) < max_vocab_size:
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word
    
    return word2idx, idx2word


# VAE Model with LSTM Encoder/Decoder and Gaussian Latent Space

class VAE(nn.Module):
    """Variational Autoencoder for text generation"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, latent_dim=32, num_layers=2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder: LSTM + projection to latent space
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: LSTM + projection to vocabulary
        self.decoder = nn.LSTM(embedding_dim + latent_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=0.1)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Regularization
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def encode(self, x):
        """Encode input sequence to latent distribution parameters"""
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.encoder(embedded)
        h = self.layer_norm(hidden[-1])  # Use last layer, apply normalization
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, target=None, max_len=25):
        """Decode latent vector to text sequence"""
        if target is not None:
            # Training mode: teacher forcing
            batch_size, seq_len = target.size()
            z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
            target_embedded = self.dropout(self.embedding(target))
            decoder_input = torch.cat([target_embedded, z_expanded], dim=2)
            lstm_out, _ = self.decoder(decoder_input)
            output = self.fc_out(lstm_out)
            return output
        else:
            # Generation mode: autoregressive
            generated = []
            current_word = torch.tensor([[1]], device=z.device)  # Start with <sos>
            
            h = torch.zeros(self.num_layers, 1, self.hidden_dim, device=z.device)
            c = torch.zeros(self.num_layers, 1, self.hidden_dim, device=z.device)
            
            for step in range(max_len):
                embedded = self.embedding(current_word)
                z_expanded = z.unsqueeze(1)
                decoder_input = torch.cat([embedded, z_expanded], dim=2)
                
                lstm_out, (h, c) = self.decoder(decoder_input, (h, c))
                output = self.fc_out(lstm_out)
                
                probs = torch.softmax(output, dim=2)
                probs[0, 0, 0] = 0  # <pad>
                probs[0, 0, 1] = 0  # <sos>
                probs = probs / probs.sum()
                
                next_word = torch.multinomial(probs.view(-1), 1).view(1, 1)
                word_idx = next_word.item()
                
                generated.append(word_idx)
                
                if word_idx == 2:  # Stop at <eos>
                    break
                
                current_word = next_word
            
            return generated
    
    def forward(self, x, target=None):
        """Forward pass: encode -> sample -> decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, target)
        return output, mu, logvar


# Word Dropout Function

def apply_word_dropout(batch, word2idx, keep_rate=0.8):
    """Randomly replace words with <unk> to force latent usage"""
    unk_idx = word2idx["<unk>"]
    pad_idx = word2idx["<pad>"]
    sos_idx = word2idx["<sos>"]
    eos_idx = word2idx["<eos>"]
    
    batch_np = batch.cpu().numpy()
    for i in range(batch_np.shape[0]):
        for j in range(batch_np.shape[1]):
            token = batch_np[i, j]
            if token not in [pad_idx, sos_idx, eos_idx]:
                if random.random() > keep_rate:
                    batch_np[i, j] = unk_idx
    return torch.tensor(batch_np, dtype=torch.long, device=batch.device)

def train_vae(model, dataloader, word2idx, epochs=300, lr=0.001):
    """Train VAE with advanced techniques"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    beta = 0.0  # Start with no KL penalty
    beta_increment = 0.001
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        total_kl = 0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Apply word dropout
            dropped_batch = apply_word_dropout(batch, word2idx, keep_rate=0.8)
            
            # Forward pass
            output, mu, logvar = model(dropped_batch, batch)
            loss = vae_loss(output, batch, mu, logvar, model.vocab_size, beta)
            
            # Monitor KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_kl += kl_div.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Learning rate scheduling
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        # Beta annealing
        beta = min(0.1, beta + beta_increment)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 30 == 0:
            avg_kl = total_kl / num_batches
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}, Loss: {avg_loss:.4f}, KL: {avg_kl:.4f}, Beta: {beta:.3f}, LR: {current_lr:.6f}')
    
    return model

# Step 4: Generation and Evaluation
def decode_sequence(indices, idx2word):
    """Convert token indices back to text"""
    words = []
    for idx in indices:
        if idx == 2:  # Stop at <eos>
            break
        if idx not in [0, 1]:  # Skip <pad> and <sos>
            words.append(idx2word[idx])
    return ' '.join(words)

def generate_text(model, idx2word, num_samples=15):
    """Generate text by sampling from prior distribution"""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim)
        sentences = []
        
        for i in range(num_samples):
            indices = model.decode(z[i:i+1])
            sentence = decode_sequence(indices, idx2word)
            sentences.append(sentence)
    
    return sentences

def interpolate_text(model, sent1, sent2, word2idx, idx2word, steps=7):
    """Interpolate between two sentences in latent space"""
    def encode_sentence(sent):
        encoded = [word2idx["<sos>"]] + [word2idx.get(w, word2idx["<unk>"]) for w in sent.split()] + [word2idx["<eos>"]]
        encoded = encoded + [word2idx["<pad>"]] * (25 - len(encoded))
        return torch.tensor([encoded], dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        z1, _ = model.encode(encode_sentence(sent1))
        z2, _ = model.encode(encode_sentence(sent2))
        interpolated_sentences = []
        
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            indices = model.decode(z)
            sentence = decode_sequence(indices, idx2word)
            interpolated_sentences.append(sentence)
    
    return interpolated_sentences

def calculate_perplexity(model, dataloader, word2idx):
    """Calculate perplexity on test set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            output, mu, logvar = model(batch, batch)
            loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')(output.view(-1, model.vocab_size), batch.view(-1))
            total_loss += loss.item()
            
            # Count non-padding tokens
            non_pad_tokens = (batch != word2idx["<pad>"]).sum().item()
            total_tokens += non_pad_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

# Step 3: Training Functions
def vae_loss(recon_x, x, mu, logvar, vocab_size, beta=0.1):
    """VAE loss: reconstruction + KL divergence"""
    recon_loss = nn.CrossEntropyLoss(ignore_index=0)(recon_x.view(-1, vocab_size), x.view(-1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# Example Usage 
if __name__ == "__main__":
    print("=== VAE Text Generation ===")
    
    # Load and preprocess data
    data_paths = ["Sarcasm_Headlines_Dataset.json", "Sarcasm_Headlines_Dataset_v2.json"]
    print("Loading datasets...")
    sentences = load_dataset(data_paths, max_sentences=3000)
    print(f"Total headlines: {len(sentences)}")
    print(f"Sample: {sentences[0]}")
    print()
    
    # Build vocabulary
    print("Building vocabulary...")
    word2idx, idx2word = build_vocabulary(sentences, min_freq=2, max_vocab_size=3000)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    print()
    
    # Create data loader
    dataset = TextDataset(sentences, word2idx, max_len=25)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset size: {len(dataset)}")
    print()
    
    # Initialize model
    model = VAE(vocab_size, embedding_dim=128, hidden_dim=256, latent_dim=32, num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train model
    print("Training VAE...")
    trained_model = train_vae(model, dataloader, word2idx, epochs=300, lr=0.001)
    print("Training completed!")
    print()
    
    # Calculate perplexity
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(trained_model, dataloader, word2idx)
    print(f"Perplexity: {perplexity:.2f}")
    print()
    
    # Generate text
    print("Generated headlines:")
    generated = generate_text(trained_model, idx2word, num_samples=15)
    for i, sent in enumerate(generated, 1):
        print(f"{i}. {sent}")
    print()
    
    # Test reconstruction
    print("Testing reconstruction:")
    trained_model.eval()
    with torch.no_grad():
        for i, sent in enumerate(sentences[:3]):
            print(f"Original: {sent}")
            encoded = [word2idx["<sos>"]] + [word2idx.get(w, word2idx["<unk>"]) for w in sent.split()] + [word2idx["<eos>"]]
            encoded = encoded + [word2idx["<pad>"]] * (25 - len(encoded))
            enc_tensor = torch.tensor([encoded], dtype=torch.long)
            mu, _ = trained_model.encode(enc_tensor)
            indices = trained_model.decode(mu)
            reconstructed = decode_sequence(indices, idx2word)
            print(f"Reconstructed: {reconstructed}")
            print()
    
    # Test interpolation
    print("Latent space interpolation:")
    if len(sentences) >= 2:
        sent1 = sentences[0]
        sent2 = sentences[1]
        print(f"From: {sent1}")
        interpolated = interpolate_text(trained_model, sent1, sent2, word2idx, idx2word, steps=7)
        for i, s in enumerate(interpolated):
            print(f"  {i+1}. {s}")
        print(f"To: {sent2}")
    print()

    print("Generated sentence:", decode_sequence(generated_indices, idx2word))
    print("Original:", sentences[0])
    print("Reconstructed:", decode_sequence(recon_indices, idx2word)) 
import torch, torch.nn as nn
from torch.nn import functional as F
import yaml

# Les données
with open('input.txt' , 'r', encoding = 'utf-8') as f:
    texte = f.read()


# Encodage/Décodage
chars = sorted(list(set(texte)))
vocab_size = len(chars)

stoi = {c : i for i, c in enumerate(chars)}
itos = {i : c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])


# Split des données
data = torch.tensor(encode(texte), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias= False)
        self.query = nn.Linear(n_embd, head_size, bias= False)
        self.value = nn.Linear(n_embd, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei =  q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, block_size, n_embd, dropout):
        super().__init__() 
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return  self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CharacterTransformer(nn.Module):

    def __init__(self ,n_embd, n_head, block_size, dropout, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
    
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, idx, targets=None, device="cuda"):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (B,T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f (x))
        
        if targets is None:
            perte = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B* T, C)
            targets = targets.view(B * T)
            perte = F.cross_entropy(logits, targets)
        
        
        return logits, perte

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B,T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] #(B, C)
            

            probs = F.softmax(logits, dim=-1) #(B, C)

            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)

            idx = torch.cat((idx, idx_next), dim = 1) #(B, T+1)
        return idx


def main(batch_size = 64,
        block_size = 256,
        max_iters = 5000,
        eval_interval = 100,
        learning_rate = 3e-4,
        device = 'cuda',
        eval_iters = 100,
        n_embd = 384,
        n_head = 8,
        n_layer = 4,
        dropout = 0.2):

    def get_batch(split):
        data_ = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_) - block_size, (batch_size,))
        x = torch.stack([data_[i : i + block_size] for i in ix])
        y = torch.stack([data_[i+1 : i + 1 + block_size] for i in ix])
        x , y = x.to(device), y.to(device)
        return x, y




    @torch.no_grad()
    def estimation_perte(model):
        out = {}
        model.eval()  

        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        model.train()
        return out


    def train():
        model = CharacterTransformer(n_embd, n_head, block_size, dropout, n_layer)
        m = model.to(device)

        with open("transformer.yaml", "w") as f :
            yaml.dump({"n_embd" :n_embd, "n_head":n_head, "block_size":block_size, "dropout":dropout, "n_layer":n_layer}, f, sort_keys=False)
        optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

        for pas in range(1, max_iters + 1):

            if pas % eval_interval == 0:
                pertes = estimation_perte(model)
                print(f"pas {pas}: perte train {pertes['train']:.4f} perte val {pertes['val']:.4f}")

            xb, yb = get_batch('train')

            logits, perte = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            perte.backward()
            optimizer.step()

            if (pas % 500) == 0:
                contexte = torch.zeros((1,1), dtype=torch.long, device=device)
                print(decode(m.generate(contexte, max_new_tokens= 500)[0].tolist()))
        
                torch.save(model.state_dict(), "transformer.pth")       
                with open('transformer.txt', 'w') as f:
                    f.write(repr(model))
            

    train()

if __name__ == "__main__":
    main()

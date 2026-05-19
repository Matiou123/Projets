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


def main(batch_size = 32,
        block_size = 128,
        max_iters = 5000,
        eval_interval = 100,
        learning_rate = 3e-4,
        device = 'cpu',
        eval_iters = 200,
        n_embd = 32,
        n_head = 4,
        n_layer = 2,
        dropout = 0.2,
        n_decision_heads= 4):

    def get_batch(split):
        data_ = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_) - block_size, (batch_size,))
        x = torch.stack([data_[i : i + block_size] for i in ix])
        y = torch.stack([data_[i+1 : i + 1 + block_size] for i in ix])
        x , y = x.to(device), y.to(device)
        return x, y


    class Head(nn.Module):

        def __init__(self, head_size):
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

        def __init__(self, num_heads, head_size):
            super().__init__() 
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(n_embd, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class FeedForward(nn.Module):

        def __init__(self, n_embd):
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

        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedForward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x


    class DecisionCharacterTransformer(nn.Module):

        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
            self.lm_head = nn.Linear(n_embd, vocab_size)

            self.alphas = nn.Parameter(torch.zeros(1, vocab_size, n_decision_heads))
            self.decision_heads = nn.ModuleList([nn.Sequential(Block(n_embd, n_head) , nn.LayerNorm(n_embd), nn.Linear(n_embd, vocab_size))for _ in range(n_decision_heads)])
            self.decision = nn.Sequential(nn.LayerNorm(vocab_size * vocab_size), nn.Linear(vocab_size*vocab_size, n_embd, bias=False),  nn.GELU(),  nn.Linear(n_embd, vocab_size, bias=False))

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (B,T,C)
            x = tok_emb + pos_emb
            x = self.blocks(x)
            cat = torch.cat([h(x).unsqueeze(3) for h in self.decision_heads ], dim = 3)
            
            logits_heads = [cat[:,:,:,i] for i in range(cat.shape[-1])]

            alphas = torch.softmax(self.alphas, dim = 2)
            scores = cat @ alphas.permute(0, 2, 1)
           
            
            logits = self.decision(scores.view(B, T, vocab_size * vocab_size))


            if targets is None:
                perte = None; perte_heads= None
            else:
                B, T, C = logits.shape
                logits = logits.view(B* T, C)
                targets = targets.view(B * T)
                perte = F.cross_entropy(logits, targets)
                
                perte_heads = sum([F.cross_entropy(l.view(B*T,C), targets) for l in logits_heads])
            
            return logits, perte, perte_heads

        @torch.no_grad()
        def generate(self, idx, max_new_tokens):
            # idx is (B,T)
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]

                logits, _ , _= self(idx_cond)

                logits = logits[:, -1, :] #(B, C)

                probs = F.softmax(logits, dim=-1) #(B, C)

                idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
    
                idx = torch.cat((idx, idx_next), dim = 1) #(B, T+1)
                
            return idx
    @torch.no_grad()
    def estimation_perte(model):
        out = {}
        model.eval()  

        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, _ , _ = model(X,Y)
                loss = F.cross_entropy(logits, Y.view(-1))
                losses[k] = loss.item()
            out[split] = losses.mean()

        model.train()
        return out


    def train():
        model = DecisionCharacterTransformer()
        m = model.to(device)


        optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

        for pas in range(max_iters):

            if pas % eval_interval == 0:
                pertes = estimation_perte(model)
                print(f"pas {pas}: perte train {pertes['train']:.4f} perte val {pertes['val']:.4f}")

            xb, yb = get_batch('train')

            logits, perte, perte_heads = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            perte += perte_heads
            perte.backward()
            optimizer.step()

            if (pas % 500) == 0:
                contexte = torch.zeros((1,1), dtype=torch.long, device=device)
                print(decode(m.generate(contexte, max_new_tokens= 500)[0].tolist()))
        
                torch.save(model.state_dict(), "décideur_transformer.pth")       
                with open('décideur_transformer.txt', 'w') as f:
                    f.write(repr(model))
            
                with open('décideur_transformer.yaml', 'w') as f:
                    config = {
                        "batch_size" : batch_size,
                        "block_size" : block_size,
                        "max_iters" : max_iters,
                        "eval_interval" : eval_interval,
                        "learning_rate" : learning_rate,
                        "device" : device,
                        "eval_iters" : eval_iters,
                        "n_embd" : n_embd,
                        "n_head" : n_head,
                        "n_layer" : n_layer,
                        "dropout" : dropout
                    }
                    yaml.dump(config, f)
    train()


if __name__ == "__main__":
    main()

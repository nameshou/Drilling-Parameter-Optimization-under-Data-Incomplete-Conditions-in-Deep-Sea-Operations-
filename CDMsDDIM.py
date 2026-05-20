import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import time
from copy import deepcopy

class FiLM(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.scale = nn.Linear(cond_dim, dim)
        self.shift = nn.Linear(cond_dim, dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.constant_(self.scale.bias, 1.0)
        nn.init.constant_(self.shift.bias, 0.0)
    
    def forward(self, x, cond):
        return x * (1 + self.scale(cond)) + self.shift(cond)


class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.film = FiLM(dim, cond_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x, cond):
        residual = x
        x = self.norm1(x)
        x = self.act(self.linear1(x))
        x = self.film(x, cond)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.act(self.linear2(x))
        return residual + x


class ImprovedDiffusionModel(nn.Module):
    def __init__(self, unknown_dim, known_dim, hidden_dim=256, num_blocks=4, dropout=0.1):
        super().__init__()
        self.unknown_dim = unknown_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(known_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_proj = nn.Linear(unknown_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, unknown_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
    
    def forward(self, x, t, condition):
        t_emb = self.time_embed(t.unsqueeze(-1) / 100.0)
        c_emb = self.cond_embed(condition)
        cond = self.fuse(torch.cat([t_emb, c_emb], dim=-1))
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x)

class ImprovedDiffusion:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t = t.long().clamp(0, self.T - 1)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, x_start


class StandardSampler:
    def __init__(self, diffusion, num_steps=50, eta=0.0):
        self.diffusion = diffusion
        self.num_steps = num_steps
        self.alphas_cumprod = diffusion.alphas_cumprod
        self.eta = eta
    
    @torch.no_grad()
    def sample(self, model, condition):
        model.eval()
        batch_size = condition.shape[0]
        latent_dim = model.unknown_dim
        device = condition.device
        
        x = torch.randn(batch_size, latent_dim, device=device)
        times = torch.linspace(self.diffusion.T - 1, 0, self.num_steps + 1).long().to(device)
        
        for i in range(self.num_steps):
            t = times[i]
            t_next = times[i + 1] if i < self.num_steps - 1 else -1
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float)
            
            pred_x0 = model(x, t_tensor, condition)
            alpha_t = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            
            eps = (x - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t + 1e-8)
            c1 = torch.sqrt(alpha_next)
            c2 = torch.sqrt(1 - alpha_next)
            x = c1 * pred_x0 + c2 * eps
        return x

class DynamicDiffusionDataset(Dataset):
    def __init__(self, data, cond_dim):
        self.data = data
        self.cond_dim = cond_dim
        self.n_features = data.shape[1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        all_idx = np.arange(self.n_features)
        known_idx = np.random.choice(all_idx, self.cond_dim, replace=False)
        unknown_idx = np.array([i for i in all_idx if i not in known_idx])
        cond = torch.FloatTensor(x[known_idx])
        target = torch.FloatTensor(x[unknown_idx])
        return cond, target

class DiffusionDrillingGenerator:
 
    def __init__(self, schemes, missing_rate=0.5, T=200, num_steps=50,
                 hidden_dim=256, num_blocks=4, dropout=0.1, device='cpu'):
      
        self.schemes = np.array(schemes, dtype=np.float32)
        self.missing_rate = missing_rate
        self.n_features = self.schemes.shape[1]
        self.cond_dim = int(self.n_features * (1 - missing_rate))
        self.unknown_dim = self.n_features - self.cond_dim
        self.device = device
     
        self.scaler = StandardScaler()
        self.schemes_scaled = self.scaler.fit_transform(self.schemes)
 
        self.diffusion = ImprovedDiffusion(T=T, device=device)
        self.num_steps = num_steps
  
        self.model = ImprovedDiffusionModel(
            self.unknown_dim, self.cond_dim,
            hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout
        ).to(device)
        
        self.sampler = None  
    
    def train(self, epochs=800, batch_size=64, lr=2e-4, test_ratio=0.2):

        n_samples = len(self.schemes_scaled)
        n_test = int(n_samples * test_ratio)
        train_data = self.schemes_scaled[:-n_test] if n_test > 0 else self.schemes_scaled
        test_data = self.schemes_scaled[-n_test:] if n_test > 0 else self.schemes_scaled
    
        train_dataset = DynamicDiffusionDataset(train_data, self.cond_dim)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
  
        ema_model = deepcopy(self.model)
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_decay = 0.999
        
        def update_ema():
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), self.model.parameters()):
                    ema_p.data = ema_decay * ema_p.data + (1 - ema_decay) * p.data

        def warmup_lambda(step):
            return min(1.0, step / 500)
        scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
        scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-500, eta_min=1e-6)

        global_step = 0
        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            for cond, target in train_loader:
                cond = cond.to(self.device)
                target = target.to(self.device)
                
                t = torch.randint(0, self.diffusion.T, (target.size(0),), device=self.device).float()
                noise = torch.randn_like(target)
                noisy_target, target_x0 = self.diffusion.q_sample(target, t, noise)
                
                pred_x0 = self.model(noisy_target, t, cond)
                loss = criterion(pred_x0, target_x0)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                update_ema()
                global_step += 1
                
                if global_step <= 500:
                    scheduler_warmup.step()
            
            if global_step > 500:
                scheduler_cos.step()
        
        self.training_time = time.time() - start_time
        self.model = ema_model  
        self.sampler = StandardSampler(self.diffusion, num_steps=self.num_steps)
        
        return self.training_time
    
    def generate(self, query, known_positions, num_samples=1):

        cond_vec = np.array([query[pos] for pos in known_positions], dtype=np.float32)

        full_template = np.zeros((1, self.n_features), dtype=np.float32)
        for i, pos in enumerate(known_positions):
            full_template[0, pos] = cond_vec[i]
        full_scaled = self.scaler.transform(full_template)
        cond_scaled = full_scaled[0, known_positions]

        cond_tensor = torch.FloatTensor(cond_scaled).unsqueeze(0).repeat(num_samples, 1).to(self.device)
        
        with torch.no_grad():
            generated_unknown = self.sampler.sample(self.model, cond_tensor).cpu().numpy()

        unknown_positions = [i for i in range(self.n_features) if i not in known_positions]
        generated_schemes = []
        for k in range(num_samples):
            full_scaled_sample = full_scaled.copy()
            for j, pos in enumerate(unknown_positions):
                full_scaled_sample[0, pos] = generated_unknown[k, j]
            full_orig = self.scaler.inverse_transform(full_scaled_sample)[0]
            generated_schemes.append(full_orig.tolist())

        scheme_ids = list(range(num_samples))  
        return scheme_ids, generated_schemes
    
    def measure_inference_time(self, query, known_positions, num_runs=100):
        cond_vec = np.array([query[pos] for pos in known_positions], dtype=np.float32)
        full_template = np.zeros((1, self.n_features), dtype=np.float32)
        for i, pos in enumerate(known_positions):
            full_template[0, pos] = cond_vec[i]
        full_scaled = self.scaler.transform(full_template)
        cond_scaled = full_scaled[0, known_positions]
        cond_tensor = torch.FloatTensor(cond_scaled).unsqueeze(0).to(self.device)
        
        for _ in range(10):
            _ = self.sampler.sample(self.model, cond_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_runs):
            _ = self.sampler.sample(self.model, cond_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        return (time.time() - start) / num_runs * 1000

if __name__ == "__main__":

    excel_path = r"E:\tool\python project\test\钻速机器学习.xlsm"
    df = pd.read_excel(excel_path, engine='openpyxl')
    schemes = df.values.astype(np.float32).tolist()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = DiffusionDrillingGenerator(
        schemes, missing_rate=5/11, T=200, num_steps=50,
        hidden_dim=256, num_blocks=4, dropout=0.1, device=device
    )

    training_time = generator.train(epochs=800, batch_size=64, lr=2e-4)
    
    known_positions = [0, 1, 2, 4, 5, 6]
    my_query = {0: 0, 1: 50, 2: 60, 4: 21, 5: 1.27, 6: 50}

    gen_ids, gen_data = generator.generate(my_query, known_positions, num_samples=10)
    print(f"\n生成 {len(gen_data)} 条差异性方案:")
    for i, (sid, data) in enumerate(zip(gen_ids, gen_data)):
        print(f"  方案{sid}: {[round(v, 2) for v in data]}")
    
    inference_time = generator.measure_inference_time(my_query, known_positions, num_runs=100)
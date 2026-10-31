# Migrated from 1002genertation.ipynb, cell 25. See docs/MIGRATION.md.
# ============================================================
# 2. Context-aware Graph AutoEncoder
# ============================================================

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.LayerNorm(out_dim), nn.Dropout(dropout))

    def forward(self, x): return self.net(x)


def initialize_rgb_prototypes(K):
    """Initialize rgb prototypes."""
    if K == 1: return np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

    i = np.arange(K, dtype=np.float32)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / (K - 1)
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    x = np.cos(golden * i) * r
    z = np.sin(golden * i) * r
    sphere = np.stack([x, y, z], axis=1)

    rgb = 0.5 + 0.28 * sphere
    return np.clip(rgb, 0.12, 0.88).astype(np.float32)


class ContextGraphAutoEncoder(nn.Module):
    """Contextgraphautoencoder."""
    def __init__(self, num_celltypes, latent_dim=3, context_hidden_dims=(128,64), decoder_hidden_dims=(64,128), eta=0.20, context_scale=1.0, dropout=0.0):
        super().__init__()
        if latent_dim != 3: raise ValueError(" the current model is RGB designed for , latent_dim must be 3.")

        self.num_celltypes = int(num_celltypes); self.latent_dim = 3
        self.eta = float(eta); self.context_scale = float(context_scale)

        init_rgb = initialize_rgb_prototypes(self.num_celltypes)
        init_logits = np.log(init_rgb / (1.0 - init_rgb))
        self.prototype_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))

        layers, prev = [], self.num_celltypes
        for h in context_hidden_dims:
            layers.append(MLPBlock(prev, int(h), dropout)); prev = int(h)
        layers.append(nn.Linear(prev, 3))
        self.context_encoder = nn.Sequential(*layers)

        layers, prev = [], 3
        for h in decoder_hidden_dims:
            layers.append(MLPBlock(prev, int(h), dropout)); prev = int(h)
        self.decoder_backbone = nn.Sequential(*layers)
        self.type_head = nn.Linear(prev, self.num_celltypes)
        self.env_head = nn.Linear(prev, self.num_celltypes)

    def encode(self, self_x, neighbor_P):
        labels = torch.argmax(self_x, dim=1)
        prototype_logits = self.prototype_logits[labels]
        delta = torch.tanh(self.context_encoder(neighbor_P))
        z = torch.sigmoid(prototype_logits + self.eta * self.context_scale * delta)
        return z, delta

    def decode(self, z):
        h = self.decoder_backbone(z)
        return self.type_head(h), self.env_head(h)

    def forward(self, self_x, neighbor_P):
        z, delta = self.encode(self_x, neighbor_P)
        type_logits, env_logits = self.decode(z)
        return type_logits, env_logits, z, delta

    def get_prototypes(self):
        return torch.sigmoid(self.prototype_logits)


# ============================================================
# 3. Loss
# ============================================================

class ContextGraphAELoss(nn.Module):
    def __init__(self, w_type=1.0, w_env=0.5, w_res=0.02, w_bg=0.01, m_bg=0.30, class_weights=None, eps=1e-8):
        super().__init__()
        self.w_type = float(w_type); self.w_env = float(w_env)
        self.w_res = float(w_res); self.w_bg = float(w_bg)
        self.m_bg = float(m_bg); self.eps = float(eps)

        if class_weights is None: self.class_weights = None
        else: self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, type_logits, env_logits, labels, target_env, z, delta):
        loss_type = F.cross_entropy(type_logits, labels, weight=self.class_weights)
        loss_env = F.kl_div(F.log_softmax(env_logits, dim=1), target_env, reduction="batchmean")
        loss_res = delta.pow(2).mean()

        dist_white = torch.sqrt(((z - 1.0) ** 2).sum(dim=1) + self.eps)
        loss_bg = F.relu(self.m_bg - dist_white).pow(2).mean()

        total = self.w_type * loss_type + self.w_env * loss_env + self.w_res * loss_res + self.w_bg * loss_bg

        with torch.no_grad():
            type_acc = (type_logits.argmax(1) == labels).float().mean()
            pred_env = F.softmax(env_logits, dim=1)
            env_cos = F.cosine_similarity(pred_env, target_env, dim=1).mean()

        return total, {
            "total": float(total.detach().cpu()),
            "type": float(loss_type.detach().cpu()),
            "env": float(loss_env.detach().cpu()),
            "res": float(loss_res.detach().cpu()),
            "bg": float(loss_bg.detach().cpu()),
            "type_acc": float(type_acc.detach().cpu()),
            "env_cos": float(env_cos.detach().cpu()),
        }


# ============================================================
# 4. Training
# ============================================================

def train_context_graph_autoencoder(
    anndata_dict, celltypes, spatial_key="spatial", celltype_key="cell_type", slice_names=None,
    knn_k=20, distance_weighted=True,
    val_fraction=0.2, batch_size=4096, num_epochs=200, lr=1e-3, weight_decay=1e-4,
    latent_dim=3, context_hidden_dims=(128,64), decoder_hidden_dims=(64,128),
    eta=0.20, context_scale=1.0, dropout=0.0,
    w_type=1.0, w_env=0.5, w_res=0.02, w_bg=0.01, m_bg=0.30, use_class_balance=True,
    early_stopping=True, early_stopping_patience=30, early_stopping_min_delta=1e-4,
    restore_best_model=True, num_workers=0, device=None, save_path=None, seed=1234,
):
    K = len(celltypes)
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    g = torch.Generator(); g.manual_seed(seed)

    dataset = ContextCellDataset(
        anndata_dict=anndata_dict, celltypes=celltypes, spatial_key=spatial_key,
        celltype_key=celltype_key, k=knn_k, distance_weighted=distance_weighted,
        slice_names=slice_names,
    )

    train_idx, val_idx = stratified_train_val_indices(dataset.labels, val_fraction=val_fraction, seed=seed)
    if len(train_idx) == 0: raise ValueError(" training set is empty .")

    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist()) if len(val_idx) > 0 else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False) if val_dataset is not None else None

    class_weights = None
    if use_class_balance:
        counts = np.bincount(dataset.labels[train_idx], minlength=K).astype(np.float32)
        class_weights = np.zeros(K, dtype=np.float32)
        present = counts > 0
        class_weights[present] = 1.0 / np.sqrt(counts[present])
        class_weights[present] /= class_weights[present].mean()

    model = ContextGraphAutoEncoder(
        num_celltypes=K, latent_dim=latent_dim,
        context_hidden_dims=context_hidden_dims, decoder_hidden_dims=decoder_hidden_dims,
        eta=eta, context_scale=context_scale, dropout=dropout,
    ).to(device)

    criterion = ContextGraphAELoss(
        w_type=w_type, w_env=w_env, w_res=w_res, w_bg=w_bg,
        m_bg=m_bg, class_weights=class_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.05)

    history = {"train": [], "val": [], "best_epoch": None, "best_val": None, "stopped": False, "stop_epoch": None}
    best_val, best_epoch, best_state, bad_epochs = float("inf"), None, None, 0

    if save_path is not None:
        save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)

    def save_ckpt(state, epoch, best_val_value=None):
        if save_path is None: return
        torch.save({
            "model_state_dict": state,
            "celltypes": list(celltypes),
            "num_celltypes": K,
            "latent_dim": latent_dim,
            "context_hidden_dims": tuple(context_hidden_dims),
            "decoder_hidden_dims": tuple(decoder_hidden_dims),
            "eta": float(eta),
            "context_scale": float(context_scale),
            "dropout": float(dropout),
            "knn_k": int(knn_k),
            "distance_weighted": bool(distance_weighted),
            "spatial_key": str(spatial_key),
            "celltype_key": str(celltype_key),
            "epoch": int(epoch),
            "best_val": None if best_val_value is None else float(best_val_value),
            "input_mode": "cell_type_plus_knn_context",
        }, save_path)

    print(f"\n[Split] total={len(dataset)}, train={len(train_idx)}, val={len(val_idx)}")
    pbar = tqdm(range(1, num_epochs + 1), desc="Train context-aware cell AE", dynamic_ncols=True)

    for epoch in pbar:
        model.train(); train_losses = []

        for batch in train_loader:
            self_x = batch["self_x"].to(device)
            neighbor_P = batch["neighbor_P"].to(device)
            labels = batch["label"].to(device)

            type_logits, env_logits, z, delta = model(self_x, neighbor_P)
            loss, loss_dict = criterion(type_logits, env_logits, labels, neighbor_P, z, delta)

            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); train_losses.append(loss_dict)

        scheduler.step()
        train_avg = average_loss_dict(train_losses); history["train"].append(train_avg)

        postfix = {"loss": f"{train_avg['total']:.4f}", "type": f"{train_avg['type']:.3f}", "env": f"{train_avg['env']:.3f}", "acc": f"{train_avg['type_acc']:.3f}", "ecos": f"{train_avg['env_cos']:.3f}"}
        should_stop = False

        if val_loader is not None:
            model.eval(); val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    self_x = batch["self_x"].to(device)
                    neighbor_P = batch["neighbor_P"].to(device)
                    labels = batch["label"].to(device)

                    type_logits, env_logits, z, delta = model(self_x, neighbor_P)
                    _, loss_dict = criterion(type_logits, env_logits, labels, neighbor_P, z, delta)
                    val_losses.append(loss_dict)

            val_avg = average_loss_dict(val_losses); history["val"].append(val_avg)
            current_val = float(val_avg["total"]); postfix["val"] = f"{current_val:.4f}"

            if current_val < best_val - float(early_stopping_min_delta):
                best_val, best_epoch, bad_epochs = current_val, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                history["best_epoch"], history["best_val"] = int(epoch), float(best_val)
                save_ckpt(best_state, epoch, best_val)
            else:
                bad_epochs += 1

            postfix["best"] = f"{best_val:.4f}"; postfix["bad"] = bad_epochs

            if early_stopping and bad_epochs >= early_stopping_patience:
                should_stop = True; history["stopped"] = True; history["stop_epoch"] = int(epoch)

        elif epoch == num_epochs:
            save_ckpt(model.state_dict(), epoch)

        pbar.set_postfix(postfix)

        if epoch % 25 == 0 or epoch == num_epochs or should_stop:
            msg = f"[Epoch {epoch:03d}] loss={train_avg['total']:.5f} type={train_avg['type']:.5f} env={train_avg['env']:.5f} acc={train_avg['type_acc']:.4f} env_cos={train_avg['env_cos']:.4f}"
            if val_loader is not None: msg += f" | val={current_val:.5f}"
            pbar.write(msg)

        if should_stop: break

    if restore_best_model and best_state is not None:
        model.load_state_dict(best_state); model.to(device)
        print(f"[Restore best] epoch={best_epoch}, val={best_val:.6f}")

    return model, history




def load_dominant_type_autoencoder(weight_path, device=None, expected_celltypes=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(weight_path, map_location=device)
    saved_celltypes = list(ckpt["celltypes"])

    if expected_celltypes is not None and list(expected_celltypes) != saved_celltypes:
        raise ValueError(" current  celltypes  order differs from the trained model .")

    model = ContextGraphAutoEncoder(
        num_celltypes=ckpt["num_celltypes"],
        latent_dim=ckpt["latent_dim"],
        context_hidden_dims=tuple(ckpt["context_hidden_dims"]),
        decoder_hidden_dims=tuple(ckpt["decoder_hidden_dims"]),
        eta=ckpt["eta"],
        context_scale=ckpt["context_scale"],
        dropout=ckpt["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    print("Loaded context-aware AE:", weight_path)
    print("K =", ckpt["num_celltypes"], "| kNN =", ckpt["knn_k"], "| eta =", ckpt["eta"])

    return model, ckpt
    
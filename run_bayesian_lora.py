import os
import json
import torch
from transformers import AutoTokenizer

from models.fudge.train_fudge_autoregressive import AutoregressiveFudgeClassifier
from preprocessing.pre_process_text import load_preprocessed_data

# pip install bayesian-lora
from bayesian_lora import calculate_kronecker_factors, model_evidence

OUT_DIR = "outputs/bayesian_lora"
CHECKPOINT = "outputs/fudge_model/fudge_classifier.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_eval_loader(texts: list[str], tokenizer: AutoTokenizer, batch_size: int = 8, max_len: int = 512):
    def collate(batch_prompts: list[str]):
        return tokenizer(
            batch_prompts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
    return collate, batch_size


def fwd_call_factory(model: AutoregressiveFudgeClassifier):
    # Returns a function(batch_inputs) -> logits [B, num_labels]
    def fwd_call(model_, batch_inputs):
        outputs = model_.base_model(
            input_ids=batch_inputs["input_ids"].to(DEVICE),
            attention_mask=batch_inputs["attention_mask"].to(DEVICE) if "attention_mask" in batch_inputs else None,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state[:, -1, :]
        logits = model_.classifier(last_hidden)
        return logits
    return fwd_call


def compute_log_likelihood(model: AutoregressiveFudgeClassifier, tokenizer: AutoTokenizer, texts: list[str], labels: list[int], batch_size: int = 16):
    model.eval()
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.base_model(**enc, return_dict=True)
            last_hidden = outputs.last_hidden_state[:, -1, :]
            logits = model.classifier(last_hidden)
            y = torch.tensor(labels[i:i + batch_size], device=DEVICE)
            loss = ce(logits, y)
        total_loss += float(loss.detach().cpu())
    return torch.tensor(-total_loss, device=DEVICE)  # log-likelihood = -NLL


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    base_model_name = ckpt["base_model_name"]
    num_labels = ckpt["num_labels"]
    lora_cfg = ckpt.get("lora_config", {"enabled": False})

    # Recreate model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoregressiveFudgeClassifier(
        model_name=base_model_name,
        num_labels=num_labels,
        lora_enabled=bool(lora_cfg.get("enabled", False)),
        lora_r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        lora_target_modules=lora_cfg.get("target_modules"),
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load data like training
    import pandas as pd
    df = pd.read_csv("data/tagmybook/data.csv")[["synopsis", "genre"]]
    data, _ = load_preprocessed_data(
        df,
        label_column="genre",
        decay_rate=0.1,
        seed=42,
        process_prefixes=False
    )

    # Subsets for speed
    eval_ds = data["validation"]
    texts = eval_ds["text"][:256]
    labels = eval_ds["labels"][:256]

    train_texts = data["train"]["text"][:1024]
    collate, bs = build_eval_loader(train_texts, tokenizer, batch_size=8)

    def train_loader():
        for i in range(0, len(train_texts), bs):
            batch_texts = train_texts[i:i + bs]
            yield collate(batch_texts)

    fwd_call = fwd_call_factory(model)

    print("Computing Kronecker factors...")
    factors = calculate_kronecker_factors(
        model=model,
        fwd_call=fwd_call,
        train_loader=train_loader(),
        n_kfac=8,
        lr_threshold=1e-5,
        modules_to_target=["lora"],
        use_tqdm=True,
    )

    print("Computing log-likelihood on eval subset...")
    log_likelihood = compute_log_likelihood(model, tokenizer, texts, labels, batch_size=16)

    # Evidence grid
    n_lora_grid = [2, 4, 8, 16]
    prior_var_grid = [0.1, 1.0, 10.0]
    n_kfac = 8

    best = None
    results = []
    print("Evaluating model evidence...")
    for n_lora in n_lora_grid:
        for prior_var in prior_var_grid:
            ev = model_evidence(
                model=model,
                log_likelihood=log_likelihood,
                factors=factors,
                n_lora=n_lora,
                n_kfac=n_kfac,
                prior_var=torch.tensor(prior_var, device=DEVICE),
            )
            ev_float = float(ev.detach().cpu())
            results.append({"n_lora": n_lora, "prior_var": prior_var, "evidence": ev_float})
            if best is None or ev_float > best["evidence"]:
                best = {"n_lora": n_lora, "prior_var": prior_var, "evidence": ev_float}

    with open(os.path.join(OUT_DIR, "evidence_results.json"), "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

    torch.save(factors, os.path.join(OUT_DIR, "kfac_factors.pt"))

    print("Bayesian LoRA done.")
    print("Best:", best)


if __name__ == "__main__":
    main()
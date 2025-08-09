import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from models.fudge.train_fudge_autoregressive import AutoregressiveFudgeClassifier
import time


class FudgeInference:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        fudge_model_path: str,
        target_class: int | str,  # Can now accept either class index or name
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Load checkpoint with all metadata
        print(f"Loading checkpoint from {fudge_model_path}")  # Debug print
        checkpoint = torch.load(
            fudge_model_path, 
            map_location=device,
            weights_only=True  # Add this to match training save
        )
        
        # Initialize FUDGE model (use saved LoRA config if present)
        lora_cfg = checkpoint.get('lora_config', {"enabled": False})
        self.fudge_model = AutoregressiveFudgeClassifier(
            model_name=checkpoint['base_model_name'],
            num_labels=checkpoint['num_labels'],
            lora_enabled=bool(lora_cfg.get('enabled', False)),
            lora_r=int(lora_cfg.get('r', 8)),
            lora_alpha=int(lora_cfg.get('alpha', 32)),
            lora_dropout=float(lora_cfg.get('dropout', 0.05)),
            lora_target_modules=lora_cfg.get('target_modules')
        ).to(device)
        
        # Load the saved weights
        self.fudge_model.load_state_dict(checkpoint['state_dict'])
        self.fudge_model.eval()
        
        # Load label mapping from checkpoint
        self.label_mapping = checkpoint['label_mapping']
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        self.num_classes = checkpoint['num_labels']
        
        # Handle target class specification (can be name or index)
        if isinstance(target_class, str):
            if target_class not in self.label_mapping:
                raise ValueError(f"Unknown class name: {target_class}. Available classes: {list(self.label_mapping.keys())}")
            self.target_class = self.label_mapping[target_class]
        else:
            if not 0 <= target_class < self.num_classes:
                raise ValueError(f"target_class must be between 0 and {self.num_classes-1}")
            self.target_class = target_class

    @property
    def target_class_name(self) -> str:
        """Get the name of the target class"""
        return self.reverse_mapping.get(self.target_class, str(self.target_class))

    def get_available_classes(self) -> dict[str, int]:
        """Return dictionary of available class names and their indices"""
        return self.label_mapping

    def get_next_token_distribution(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Get next token distribution from base model with sampling methods."""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            return torch.softmax(logits, dim=-1)

    def generate_with_fudge(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        base_top_k: Optional[int] = 50,     
        fudge_top_k: int = 200,             
        lambda_weight: float = 0.5
    ) -> str:
        """
        Generate text using FUDGE (Future Discriminators for Generation) algorithm.
        
        The algorithm works by:
        1. Getting next-token distribution P(x_t|x_<t) from base LM
        2. For top-k candidates, computing P(attribute|x_≤t) using FUDGE classifier
        3. Combining distributions: P(x_t|x_<t, attribute) ∝ P(x_t|x_<t) * P(attribute|x_≤t)^λ
        
        Args:
            prompt: Input text to condition generation on
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature for base model
            base_top_k: Number of tokens to consider in final sampling
            fudge_top_k: Number of candidates to evaluate with FUDGE
            lambda_weight: Weight for FUDGE scores (higher = stronger attribute control)
        
        Returns:
            Generated text string including prompt
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = input_ids.clone()
        
        print(f"Starting generation with prompt length: {len(input_ids[0])} tokens")
        print(f"Will generate up to {max_length} additional tokens")
        
        for i in range(max_length):
            # Time both operations and print in one line
            t0 = time.time()
            base_distribution = self.get_next_token_distribution(
                generated_tokens,
                temperature=temperature,
                top_k=None,
            )
            
            # Get top-k candidates from base model
            candidate_values, candidate_indices = torch.topk(
                base_distribution[0], 
                k=min(fudge_top_k, base_distribution.size(-1))
            )
            t1 = time.time()
            
            fudge_scores = []
            for token_id in candidate_indices:
                candidate_sequence = torch.cat([
                    generated_tokens, 
                    token_id.unsqueeze(0).unsqueeze(0)
                ], dim=-1)
                
                with torch.no_grad():
                    fudge_outputs = self.fudge_model(candidate_sequence)
                    fudge_probs = torch.softmax(fudge_outputs['logits'], dim=-1)[0]
                    fudge_prob = fudge_probs[self.target_class]
                    fudge_scores.append(fudge_prob)
            t2 = time.time()
            
            print(f"Token [{i+1}/{max_length}] - Base: {t1-t0:.3f}s, FUDGE: {t2-t1:.3f}s, Total: {t2-t0:.3f}s")

            # Convert to tensor and combine scores
            fudge_scores = torch.tensor(fudge_scores, device=self.device)
            
            # P(x_t|x_<t, attribute) ∝ P(x_t|x_<t) * P(attribute|x_≤t)^λ
            combined_scores = candidate_values * (fudge_scores ** lambda_weight)
            
            # Normalise
            combined_scores = combined_scores / combined_scores.sum()
            
            # Create full distribution for sampling
            combined_distribution = torch.zeros_like(base_distribution)
            combined_distribution[0, candidate_indices] = combined_scores

            # Sample next token
            next_token = torch.multinomial(combined_distribution[0], num_samples=1).unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                print(f"Generation stopped early at token {i+1} (EOS token)")
                break

        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
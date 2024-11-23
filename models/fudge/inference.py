import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class FudgeInference:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        fudge_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize FUDGE inference with pre-loaded models.
        
        Args:
            model: HuggingFace language model
            tokenizer: HuggingFace tokenizer
            fudge_model_path: Path to trained FUDGE model
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Load saved FUDGE model state
        checkpoint = torch.load(fudge_model_path, map_location=device)
        
        # Initialize FUDGE model with saved config
        from models.fudge.fudge_train_autoregressive import AutoregressiveFudgeClassifier
        self.fudge_model = AutoregressiveFudgeClassifier(
            model_name=checkpoint['base_model_name'],
            num_labels=checkpoint['num_labels']
        ).to(device)
        
        # Load the saved weights
        self.fudge_model.load_state_dict(checkpoint['state_dict'])
        self.fudge_model.eval()  # Set to evaluation mode

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
        fudge_temperature: float = 1.0,
        base_top_k: Optional[int] = 50,     # For base model sampling diversity
        base_top_p: Optional[float] = None,  # For base model nucleus sampling
        fudge_top_k: int = 200,             # Number of candidates to evaluate with FUDGE
        lambda_weight: float = 0.5
    ) -> str:
        """
        Generate text using the base model guided by FUDGE predictions.
        
        Args:
            prompt: Initial text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature for base model
            fudge_temperature: Temperature for FUDGE logits
            base_top_k: Top-k sampling parameter for base model
            base_top_p: Nucleus sampling parameter for base model
            fudge_top_k: Top-k sampling parameter for FUDGE
            lambda_weight: Weight for combining base and FUDGE distributions
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = input_ids.clone()

        for _ in range(max_length):
            # Get base model distribution (without top-k yet)
            base_distribution = self.get_next_token_distribution(
                generated_tokens,
                temperature=temperature,
                top_k=None,           # We'll apply base_top_k later
                top_p=base_top_p
            )

            # Get fudge_top_k most likely tokens to evaluate with FUDGE
            candidate_values, candidate_indices = torch.topk(
                base_distribution[0], 
                k=min(fudge_top_k, base_distribution.size(-1))
            )
            
            # Get FUDGE predictions for candidates
            fudge_scores = []
            for token_id in candidate_indices:
                candidate_sequence = torch.cat([
                    generated_tokens, 
                    token_id.unsqueeze(0).unsqueeze(0)
                ], dim=-1)
                
                with torch.no_grad():
                    fudge_outputs = self.fudge_model(candidate_sequence)
                    fudge_score = fudge_outputs['logits'][0, 1]
                    fudge_scores.append(fudge_score)
            
            # Create combined distribution over candidates
            fudge_scores = torch.tensor(fudge_scores, device=self.device)
            fudge_distribution = torch.softmax(fudge_scores / fudge_temperature, dim=-1)
            combined_scores = (1 - lambda_weight) * candidate_values + lambda_weight * fudge_distribution

            # Create full distribution and apply base model top-k
            combined_distribution = torch.zeros_like(base_distribution)
            combined_distribution[0, candidate_indices] = combined_scores
            
            if base_top_k is not None:
                top_k_values, top_k_indices = torch.topk(
                    combined_distribution[0], 
                    k=min(base_top_k, combined_distribution.size(-1))
                )
                combined_distribution = torch.zeros_like(base_distribution)
                combined_distribution[0, top_k_indices] = top_k_values

            # Sample next token
            next_token = torch.multinomial(combined_distribution[0], num_samples=1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
import sys
import time
from pathlib import Path
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.fudge.inference_fudge import FudgeInference

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear GPU memory between runs"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_inference_batch(
    prompts: list[str], 
    target_class: int | str,  # Can now use either class index or name
    base_model_name: str = "Qwen/Qwen1.5-0.5B", 
    **kwargs
):
    """Run inference on multiple prompts using same loaded models"""
    start_time = time.time()
    clear_gpu_memory()
    
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    logger.info(f"Loading FUDGE model")
    inferencer = FudgeInference(
        model=base_model,
        tokenizer=tokenizer,
        fudge_model_path=kwargs.get('fudge_model_dir', "outputs/fudge_model") + "/fudge_classifier.pt",
        target_class=target_class
    )
    
    # Print available classes
    logger.info("Available classes:")
    for class_name, class_idx in inferencer.get_available_classes().items():
        logger.info(f"  {class_idx}: {class_name}")
    logger.info(f"Conditioning on: {inferencer.target_class_name} (index: {inferencer.target_class})")
    
    logger.info(f"Starting inference on {len(prompts)} prompts")
    outputs = []
    for i, prompt in enumerate(prompts, 1):
        prompt_start = time.time()
        logger.info(f"Processing prompt {i}/{len(prompts)}")
        
        output = inferencer.generate_with_fudge(
            prompt=prompt,
            max_length=kwargs.get('max_length', 100),
            lambda_weight=kwargs.get('lambda_weight', 0.5)
        )
        outputs.append(output)
        
        prompt_time = time.time() - prompt_start
        logger.info(f"Prompt {i} completed in {prompt_time:.2f}s")
    
    # Clean up
    logger.info("Cleaning up models and memory")
    del inferencer
    del base_model
    clear_gpu_memory()
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f}s")
    
    return outputs

if __name__ == "__main__":
    test_prompts = [
        "write a concise book synopsis:"
    ]

    target_class = "horror"

    # Can use either class name or index
    outputs = run_inference_batch(
        prompts=test_prompts,
        target_class=target_class,
        base_model_name="Qwen/Qwen1.5-0.5B",
        lambda_weight=0.8
    )
    
    logger.info("\n=== Results ===")
    for i, (prompt, output) in enumerate(zip(test_prompts, outputs), 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated (conditioning on class {target_class}): {output}")
        logger.info("-" * 50)
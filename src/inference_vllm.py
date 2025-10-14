import os
import torch
import argparse
import math
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional

from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from tqdm import tqdm
from utils import load_vllm_model_and_tokenizer, get_model_path
from utils import read_row, formatInp, setup_workspace_directories, is_runpod_environment, get_model_cache_kwargs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_device(0)
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
else:
    raise RuntimeError("CUDA is required for this script")

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Target device: {DEVICE}")


def create_sampling_params(args: Dict[str, Any]) -> SamplingParams:
    """
    Create vLLM sampling parameters from arguments.
    
    Args:
        args: Configuration arguments
        
    Returns:
        vLLM SamplingParams object
    """
    if args['do_sample_decode']:
        return SamplingParams(
            temperature=args['temperature'],
            top_p=args['top_p'],
            max_tokens=args['max_len'], 
            n=1
        )
    else:
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args['max_len'],
            n=1
        )


def extract_model_output(
    full_output: str,
    model_type: str,
    input_prompt: str = ""
) -> str:
    """
    Extract the model's response from the full generated text.
    
    Args:
        full_output: Complete generated text
        model_type: Type of model used
        input_prompt: Original input prompt (for llama3)
        
    Returns:
        Extracted model response
    """

    return full_output.strip()


def infer_vllm(
    vllm_model: LLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any]
) -> None:
    """
    Run inference on evaluation data using vLLM.
    
    Args:
        vllm_model: The vLLM model
        tokenizer: The tokenizer
        eval_data: List of evaluation data points
        args: Configuration arguments
    """
    # Remove existing output file
    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])
    
    # Create sampling parameters
    sampling_params = create_sampling_params(args)
    
    logger.info('Starting vLLM inference...')
    start_time = time.time()
    
    # Process data in batches for better efficiency
    batch_size = args.get('batch_size', 1)
    total_batches = (len(eval_data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(eval_data))
        batch_data = eval_data[start_idx:end_idx]
        
        # Prepare prompts for the batch
        prompts = []
        for data_point in batch_data:
            prompt = formatInp(
                data_point,
                model=args['model'],
                use_template=args['use_template'],
                tokenizer=tokenizer
            )
            if args['add_reasoning_step']:
                prompt = prompt + " " + data_point['context_text']
            if args['infer_on_perturbed_step']:
                prompt = prompt + " " + data_point['perturb_context_text']+'.\n'+data_point['initial_step']
            if args.get('custom_prompt', ''):
                prompt = prompt + " " + args['custom_prompt']
            if args['omit_thinking']:
                prompt = prompt + "\n\n</think>"
            prompts.append(prompt)
        
        # Generate responses using vLLM
        outputs = vllm_model.generate(prompts, sampling_params)
        
        # Process and save results
        for i, output in enumerate(outputs):
            data_idx = start_idx + i
            if data_idx >= len(eval_data):
                break
                
            generated_text = output.outputs[0].text
            logger.info(f'Generated text: {generated_text}')
            with open(args['output_file_name'], 'a') as f:
                if args['use_jb']:
                    response = extract_model_output(generated_text, args['model'])
                    eval_data[data_idx] = {'prompt': eval_data[data_idx], 'response': response}
                else:
                    # For vLLM, we need to handle the output differently since it doesn't include the input
                    eval_data[data_idx]['ori_output'] = generated_text.strip()
                
                # Note: vLLM doesn't provide probability scores in the same way as transformers
                # We'll add a placeholder for compatibility
                eval_data[data_idx]['probs'] = []
                json.dump(eval_data[data_idx], f)
                f.write('\n')
    
    end_time = time.time()
    logger.info(f'vLLM inference completed in {end_time - start_time:.2f} seconds')


def infer_vllm_sequential(
    vllm_model: LLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any]
) -> None:
    """
    Run inference on evaluation data using vLLM with sequential processing for compatibility.
    
    Args:
        vllm_model: The vLLM model
        tokenizer: The tokenizer
        eval_data: List of evaluation data points
        args: Configuration arguments
    """
    # Remove existing output file
    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])
    
    # Create sampling parameters
    sampling_params = create_sampling_params(args)
    
    logger.info('Starting vLLM sequential inference...')
    start_time = time.time()
    
    for i in tqdm(range(len(eval_data)), desc="Processing samples"):
        data_point = eval_data[i]
        input_prompt = formatInp(
            data_point,
            model=args['model'],
            use_template=args['use_template'],
            tokenizer=tokenizer
        )
        if args['add_reasoning_step']:
            input_prompt = input_prompt + " " + data_point['context_text']
        # Append custom prompt if provided
        if args['infer_on_perturbed_step']:
            input_prompt = input_prompt + " " + data_point['perturb_context_text']+'.\n'+data_point['initial_step']
        #if args.'custom_prompt', ''):
        if args['omit_thinking']:
            input_prompt = input_prompt + "\n\n</think>"
        input_prompt = input_prompt + " " + args['custom_prompt']
        
        # Generate response using vLLM
        outputs = vllm_model.generate([input_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        #logger.info(f'Generated text: {generated_text}')
        # Process and save results
        with open(args['output_file_name'], 'a') as f:
            if args['use_jb']:
                response = extract_model_output(generated_text, args['model'])
                eval_data[i] = {'prompt': eval_data[i], 'response': response}
            else:
                eval_data[i]['ori_output'] = generated_text.strip()
            
            # Note: vLLM doesn't provide probability scores in the same way as transformers
            # We'll add a placeholder for compatibility
            eval_data[i]['probs'] = []
            json.dump(eval_data[i], f)
            f.write('\n')
        
        logger.debug(f'Output: {generated_text[:100]}...')
    
    end_time = time.time()
    logger.info(f'vLLM sequential inference completed in {end_time - start_time:.2f} seconds')


def main():
    """Main function to run the vLLM inference script."""
    parser = argparse.ArgumentParser(description="vLLM-based language model inference script")
    
    # Model configuration
    parser.add_argument("--model", default='llama3', type=str, help="Model type (llama, llama3, qwen, olmo)")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size (7b, 8b, 13b, 14b, 32b)")
    parser.add_argument("--peft_pth_ckpt", default='', type=str, help="Path to PEFT checkpoint (not supported with vLLM)")
    parser.add_argument("--load_ckpt", default=0, type=int, help="Whether to load checkpoint (not supported with vLLM)")
    
    # vLLM specific configuration
    parser.add_argument("--tensor_parallel_size", default=1, type=int, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", default=0.9, type=float, help="GPU memory utilization ratio")
    parser.add_argument("--max_model_len", default=None, type=int, help="Maximum model length")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for processing")
    parser.add_argument("--use_batching", default=1, type=int, help="Use batch processing for efficiency")
    
    # Data configuration
    parser.add_argument("--input", default='data/medcq.json', type=str, help="Input file path")
    parser.add_argument("--output_file_name", default='data/medcq.json', type=str, help="Output file path")
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=10, type=int, help='Right index for data slicing')
    
    # Generation configuration
   
    parser.add_argument("--temperature", default=0, type=float, help="Generation temperature")
    parser.add_argument("--top_p", default=0, type=float, help="Top-p sampling parameter")
    parser.add_argument("--do_sample_decode", default=0, type=int, help="Use sampling for decoding")
    parser.add_argument("--record_prob_max_pos", default=0, type=int, help="Max positions to record probabilities (not supported with vLLM)")
    parser.add_argument("--max_len", default=None, type=int, help="Maximum generation length")
    # Prompt configuration
    parser.add_argument("--use_jb", default=0, type=int, help="Use jailbroken prompt")
    parser.add_argument("--use_adv_suffix", default=0, type=int, help="Use adversarial suffix")
    parser.add_argument("--use_sys_prompt", default=0, type=int, help="Use system prompt")
    parser.add_argument("--use_template", default=0, type=int, help="Use default prompting template")
    parser.add_argument("--custom_prompt", default='\n The final answer is \\boxed{', type=str, help="Custom prompt to append after template formatting")
    parser.add_argument("--do_not_use_last_inst_tok", default=0, type=int, help="Don'tcustom_prompt use last instruction token")
    parser.add_argument("--use_inversion", default=0, type=int, help="Use inversion")
    parser.add_argument("--inversion_prompt_idx", default=0, type=int, help="Inversion prompt index")
    parser.add_argument("--add_reasoning_step", default=0, type=int, help="Add reasoning step")
    parser.add_argument("--infer_on_perturbed_step", default=0, type=int, help="Infer on perturbed step")
    parser.add_argument("--omit_thinking", default=0, type=int, help="Omit thinking")
    args = parser.parse_args()
    params = vars(args)
    
    # Warn about unsupported features
    if params['load_ckpt'] or params['peft_pth_ckpt']:
        logger.warning("PEFT checkpoints are not supported with vLLM. Ignoring checkpoint loading.")
    
    if params['record_prob_max_pos'] > 0:
        logger.warning("Probability recording is not supported with vLLM. Ignoring record_prob_max_pos.")
    
    logger.info(f'Model: {params["model"]}')
    logger.info(f'Using vLLM for inference')
    logger.info(f'Custom prompt: {params["custom_prompt"]}')
    # Log environment information
    if is_runpod_environment():
        logger.info("Running in RunPod environment - models will be downloaded to /workspace")
    else:
        logger.info("Running in local environment")
    
    # Load vLLM model and tokenizer
    vllm_model, tokenizer = load_vllm_model_and_tokenizer(
        params['model'],
        params['model_size'],
        params['tensor_parallel_size'],
        params['gpu_memory_utilization'],
        params['max_model_len']
    )
    if params['max_len'] is None:
        #params['max_len'] = params['max_model_len']
        logger.info(f"max len is not set, using max_model_len")
    
    # Load and preprocess data
    test_data = read_row(params['input'])
    
    # Validate indices
    if params['left'] < 0:
        params['left'] = 0
    if params['right'] > len(test_data):
        params['right'] = len(test_data)
    
    if params['left'] >= params['right']:
        raise ValueError("Left index must be less than right index")
    
    # Filter data
    test_data = [
        d for d in test_data[params['left']:params['right']]
        if 'sample_rounds' not in d or d['sample_rounds'] != 'Failed'
    ]
    

    # Run inference
    if params['use_batching'] and params['batch_size'] > 1:
        infer_vllm(vllm_model, tokenizer, test_data, params)
    else:
        infer_vllm_sequential(vllm_model, tokenizer, test_data, params)


if __name__ == "__main__":
    main() 
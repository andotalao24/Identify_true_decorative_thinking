import os
import torch
import argparse
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Memory optimization for multi-GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Suppress Mamba 2 SSM availability check
import transformers.utils.import_utils as iu
iu.is_mamba_2_ssm_available = lambda: False

from utils import (
    formatInp,
    get_model_cache_kwargs,
    get_model_path,
    is_runpod_environment,
    read_row,
    setup_workspace_directories,
)
from utils_eval import (
    compare_answers,
    perturb_chunks_range,
    randomly_replace_numbers,
    remove_boxed_answers,
    replace_all_tokens_with_ellipsis,
)

# Try to import vLLM for optional acceleration
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# Configure logging - reduce verbosity for faster startup
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration - simplified
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    logger.info(f"Using {torch.cuda.device_count()} GPU(s)")
else:
    raise RuntimeError("CUDA is required for this script")

# Global caches for workspace directories and model kwargs
_workspace_cache: Optional[Tuple[str, str]] = None
_model_cache_kwargs: Optional[Dict[str, Any]] = None

def get_cached_workspace_dirs() -> Tuple[str, str]:
    """Get cached workspace directories to avoid repeated setup.
    
    Returns:
        Tuple of (cache_dir, models_dir) paths
    """
    global _workspace_cache
    if _workspace_cache is None:
        _workspace_cache = setup_workspace_directories()
    return _workspace_cache


def get_cached_model_kwargs() -> Dict[str, Any]:
    """Get cached model kwargs to avoid repeated setup.
    
    Returns:
        Dictionary of model loading arguments
    """
    global _model_cache_kwargs
    if _model_cache_kwargs is None:
        workspace_cache_dir, workspace_models_dir = get_cached_workspace_dirs()
        
        # Only clear wrong cache locations once
        if is_runpod_environment():
            from utils import clear_wrong_cache_locations
            clear_wrong_cache_locations()
        
        _model_cache_kwargs = {
            'cache_dir': workspace_models_dir,
            'local_files_only': False,
            'force_download': False,
            'resume_download': True,
            'proxies': None,
        }
        
        if is_runpod_environment():
            _model_cache_kwargs.update({
                'use_auth_token': None,
                'mirror': None,
            })
    return _model_cache_kwargs

def load_model_and_tokenizer_optimized(
    model_type: str, 
    model_size: str, 
    use_vllm: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Optimized model loading with reduced overhead.
    
    Args:
        model_type: Type of model to load
        model_size: Size of model to load
        use_vllm: Whether vLLM is being used (affects device mapping)
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        Exception: If model loading fails
    """
    try:
        model_path = get_model_path(model_type, model_size)
        cache_kwargs = get_cached_model_kwargs()
        
        logger.info(f"Loading model: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            **cache_kwargs
        )
        
        # Load model with optimized device mapping
        if torch.cuda.device_count() > 1 and use_vllm:
            device_map = "cuda:0"
        else:
            device_map = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            **cache_kwargs
        )
        
        logger.info(f"Successfully loaded model: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_type} {model_size}: {e}")
        raise

def load_vllm_model_and_tokenizer_optimized(
    model_type: str, 
    model_size: str, 
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    gpu_devices: Optional[str] = None
) -> Tuple[LLM, AutoTokenizer]:
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not available. Install vLLM to use this function.")
    
    try:
        model_path = get_model_path(model_type, model_size)
        cache_kwargs = get_cached_model_kwargs()
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **cache_kwargs
        )
        
        # Get model configuration
        config = AutoConfig.from_pretrained(
            model_path, 
            trust_remote_code=trust_remote_code,
            **cache_kwargs
        )
        
        # Determine max_model_len efficiently
        if max_model_len is None:
            if hasattr(config, 'max_position_embeddings'):
                max_model_len = config.max_position_embeddings
            elif hasattr(config, 'max_sequence_length'):
                max_model_len = config.max_sequence_length
            elif hasattr(config, 'context_length'):
                max_model_len = config.context_length
            elif hasattr(config, 'seq_length'):
                max_model_len = config.seq_length
            elif hasattr(config, 'n_positions'):
                max_model_len = config.n_positions
            else:
                # Use model-specific defaults
                if model_type == 'olmo':
                    max_model_len = 32768 if model_size == '32b' else 8192
                elif model_type == 'llama3':
                    max_model_len = 8192
                elif model_type == 'llama':
                    max_model_len = 4096
                elif model_type in ['qwen', 'deepseek-qwen']:
                    max_model_len = 32768
                else:
                    max_model_len = 4096
        
        logger.info(f"Loading vLLM model: {model_path}")
        
        # Load vLLM model
        vllm_kwargs = {
            'model': model_path,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_model_len': max_model_len,
            'trust_remote_code': trust_remote_code,
            'dtype': "auto",
            'enforce_eager': False,
        }
        
        # Handle GPU device selection
        original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if gpu_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
        
        vllm_model = None
        try:
            vllm_model = LLM(**vllm_kwargs)
        finally:
            if original_cuda_visible_devices is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
        
        logger.info(f"Successfully loaded vLLM model: {model_path}")
        return vllm_model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load vLLM model {model_type} {model_size}: {e}")
        raise


def _get_max_model_length(
    config: AutoConfig, 
    model_type: str, 
    model_size: str
) -> int:
    """Get maximum model length from config or use defaults.
    
    Args:
        config: Model configuration
        model_type: Type of model
        model_size: Size of model
        
    Returns:
        Maximum model length
    """
    # Try different possible attribute names for max length
    for attr_name in ['max_position_embeddings', 'max_sequence_length', 
                     'context_length', 'seq_length', 'n_positions']:
        if hasattr(config, attr_name):
            return getattr(config, attr_name)
    
    # Use model-specific defaults
    if model_type == 'olmo':
        return 32768 if model_size == '32b' else 8192
    elif model_type == 'llama3':
        return 8192
    elif model_type == 'llama':
        return 4096
    elif model_type in ['qwen', 'deepseek-qwen']:
        return 32768
    else:
        return 4096




def generate_with_vllm(vllm_model: LLM, prompt: str, max_tokens: int) -> str:
    """Generate text using vLLM for faster inference.
    
    Args:
        vllm_model: The vLLM model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not available. Install vLLM to use this function.")
    
    # Create sampling parameters for greedy decoding
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        n=1
    )
    
    # Generate using vLLM
    outputs = vllm_model.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return generated_text


def clear_gpu_cache() -> None:
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")


def log_gpu_memory() -> None:
    """Quick GPU memory log - only log if significant memory usage."""
    if torch.cuda.is_available():
        total_allocated = 0
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total_allocated += allocated
        # Only log if using more than 1GB total
        if total_allocated > 1.0:
            logger.info(f"GPU memory: {total_allocated:.1f}GB total")

def generate_and_analyze_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    ans: str,
    checkpoint_idx: int,
    top_k: int,
    store_logits: bool,
    store_distribution: bool,
    max_new_tokens: int = 20
) -> Dict[str, Any]:
    """Generate and analyze a single checkpoint.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        ans: Expected answer
        checkpoint_idx: Checkpoint index
        top_k: Number of top predictions
        store_logits: Whether to store logits
        store_distribution: Whether to store distribution
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Analysis results dictionary containing:
        - checkpoint_idx: Index of the checkpoint
        - input_length: Length of input tokens
        - ans_probability: Probability of ground truth answer
        - ans_token_id: Token IDs of ground truth answer
        - generation_sequence: Sequence of generated tokens
        - extracted_generated_answer: Generated answer text
        - extracted_generated_answer_probability: Probability of generated answer
        - ans_probability_multi: Multiplicative probability of ground truth
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    
    generated_tokens = []
    generated_token_ids = []
    generated_probabilities = []
    current_input_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()
    predicted_logits = []
    
    # Track answer probability distribution
    ans_probability_distribution = []
    
    try:
        ans_probability = []
        ans_logit = []
        generated_logit = []
        ans_token_id = tokenizer.encode(ans, add_special_tokens=False)
        ans_token_text = tokenizer.decode(ans_token_id, skip_special_tokens=True)
        # Fix tokenization issue for specific model variants
        #if ans_token_id[0] == 12:
            #ans_token_id[0] = 481  # hard code for dpsk-qwen on amc, diff between '-' and ' -' 
        len_ans = len(ans_token_id)
        distribution_all = None  # Initialize as None, will be set in first iteration
        with torch.no_grad():
            ans_idx = 0
            for gen_step in range(max_new_tokens):
                outputs = model(current_input_ids, attention_mask=current_attention_mask)
                logits = outputs.logits[0, -1, :]
                #predicted_logits.append(logits)
                
                try:
                    distribution = torch.softmax(logits, dim=-1)
                except (RuntimeError, ValueError):
                    # Add numerical stability by subtracting max logit
                    logits_stable = logits - torch.max(logits)
                    distribution = torch.softmax(logits_stable, dim=-1)
                
                # Get next token using greedy decoding
                next_token_id = torch.argmax(distribution).item()
                next_token_logit = logits[next_token_id].item()
                next_token_prob = distribution[next_token_id].item()
                next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                
                # Track ground truth answer probability
                if ans_idx < len_ans:
                    ans_probability.append(distribution[ans_token_id[ans_idx]].item())
                    ans_idx += 1
                
                # Stop generation if closing brace is reached
                if next_token_text.strip() == '}' or '}' in next_token_text:
                    break
                
                # Accumulate distributions for analysis
                if distribution_all is None:
                    distribution_all = distribution.clone().cpu()
                else:
                    distribution_all += distribution.cpu()
                
                # Store generated token information
                generated_tokens.append(next_token_text)
                generated_token_ids.append(next_token_id)
                generated_probabilities.append(next_token_prob)
                # Update input for next iteration
                new_token_tensor = torch.tensor([[next_token_id]], device=DEVICE)
                current_input_ids = torch.cat([current_input_ids, new_token_tensor], dim=1)
                new_attention = torch.ones((1, 1), device=DEVICE)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
        
        # Note: Top-k predictions could be computed here if needed
        # Currently using greedy decoding, so top-1 predictions are stored
        
 
        # Combine generated tokens into full answer
        full_generated_text = ''.join(generated_tokens) if generated_tokens else ''
        
        # Build generation sequence with probabilities
        generation_sequence = []
        prob_multi_all = 1.0
        for i, (token, token_id, prob) in enumerate(zip(generated_tokens, generated_token_ids, generated_probabilities)):
            generation_sequence.append({
                'step': i + 1,
                'token_id': token_id,
                'token_text': token,
                'probability': prob
            })
            prob_multi_all *= prob
        
        # Calculate probabilities for generated and ground truth answers
        extracted_generated_answer_probability = prob_multi_all
        ans_probability_multi = 1.0
        for p in ans_probability:
            ans_probability_multi *= p
        
        # Store analysis results
        analysis = {
            'checkpoint_idx': checkpoint_idx,
            'text_up_to_checkpoint': '',  # Will be set by caller
            'input_length': input_ids.shape[1],
            'ans_probability': ans_probability,  # Ground truth answer probabilities
            'ans_token_id': ans_token_id,  # Ground truth answer token IDs
            'ans_token_sequence': ans_token_text,
            'generation_sequence': generation_sequence,
            'extracted_generated_answer': full_generated_text,
            'extracted_generated_answer_probability': extracted_generated_answer_probability,
            'ans_probability_multi': ans_probability_multi,
        }
        
        # Conditionally add logits and distribution data
        if store_logits:
            analysis['logits'] = logits.cpu().numpy().tolist()
        if store_distribution:
            # TODO: Implement distribution storage if needed
            pass
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in generation for checkpoint {checkpoint_idx}: {e}")
        return {
            'checkpoint_idx': checkpoint_idx,
            'text_up_to_checkpoint': '',
            'input_length': input_ids.shape[1] if 'input_ids' in locals() else 0,
            'error': 'generation_error',
            'error_details': str(e)
        }

def analyze_checkpoints_for_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any],
    vllm_model: Optional[LLM] = None
) -> None:
    """Analyze checkpoint predictions for a dataset with OOM handling.
    
    This function processes a dataset of evaluation examples, analyzing model behavior
    at different checkpoints during reasoning. It supports various perturbation strategies
    and two-stage generation using vLLM for improved performance.
    
    Args:
        model: The language model for analysis
        tokenizer: The tokenizer for text processing
        eval_data: List of evaluation data points
        args: Configuration arguments including:
            - output_file_name: Path to output file
            - delimiter: Text delimiter for chunking
            - random_number_replacement: Whether to use number perturbation
            - top_k: Number of top predictions to store
            - store_logits: Whether to store logit values
            - store_distribution: Whether to store probability distributions
        vllm_model: Optional vLLM model for faster stage 1 generation
    """
    max_new_tokens_stage1 = 1000  # More tokens for full reasoning
    
    # Clean up existing output files
    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])
    
    # Remove existing answer change tracking files if random replacement is enabled
    if args.get('random_number_replacement', False):
        answer_change_file = args['output_file_name'].replace('.jsonl', '_answer_changes.jsonl')
        if os.path.exists(answer_change_file):
            os.remove(answer_change_file)
        
        stage1_text_file = args['output_file_name'].replace('.jsonl', '_stage1_texts.jsonl')
        if os.path.exists(stage1_text_file):
            os.remove(stage1_text_file)
    
    # Extract configuration parameters
    delimiter = args.get('delimiter', '.')
    user_tag = "\n The final answer is \\boxed{"
    use_final_result_tag = args.get('use_final_result_tag', False)
    top_k = args.get('top_k', 10)
    store_logits = args.get('store_logits', True)
    store_distribution = args.get('store_distribution', True)
    random_number_replacement = args.get('random_number_replacement', False)
    perturb_forward_all = args.get('perturb_forward_all', False)
    
    # Validate arguments that require random_number_replacement
    if perturb_forward_all and not random_number_replacement:
        raise ValueError("--perturb_forward_all requires --random_number_replacement to be enabled")
    
    replace_all_numbers = args.get('replace_all_numbers', True)
    rephrase_with_gpt = args.get('rephrase_with_gpt', False)
    remove_boxed_answers_flag = args.get('remove_boxed_answers', False)
    
    if replace_all_numbers and not random_number_replacement:
        raise ValueError("--replace_all_numbers requires --random_number_replacement to be enabled")
    
    if rephrase_with_gpt and not random_number_replacement:
        raise ValueError("--rephrase_with_gpt requires --random_number_replacement to be enabled")
    
    logger.info(f'Starting checkpoint analysis with delimiter: "{delimiter}"')
    logger.info(f'User tag: "{user_tag}"')
    logger.info(f'Use final result tag: {use_final_result_tag}')
    logger.info(f'Store logits: {store_logits}')
    logger.info(f'Store distribution: {store_distribution}')
    logger.info(f'Random number replacement: {random_number_replacement}')
    logger.info(f'Replace number mode: {args.get("replace_number_mode", "random")}')
    logger.info(f'Perturb forward all: {perturb_forward_all}')
    logger.info(f'Replace all numbers: {replace_all_numbers}')
    logger.info(f'Rephrase with GPT: {rephrase_with_gpt}')
    logger.info(f'Remove boxed answers: {remove_boxed_answers_flag}')
    
    # Start timing
    start_time = time.time()
    
    # Track skipped samples due to OOM
    skipped_samples = []
    
    # Progress tracking for periodic saves
    progress_file = args['output_file_name'].replace('.jsonl', '_progress.json')
    processed_samples = 0
    
    for i in tqdm(range(len(eval_data)), desc="Processing samples"):
        data_point = eval_data[i]
        
        # Check if we have ori_output to analyze
        if 'ori_output' not in data_point:
            logger.warning(f"Sample {i} missing 'ori_output', skipping...")
            skipped_samples.append({'index': i, 'reason': 'missing_ori_output'})
            continue
        
        # Get ori_output text
        ori_output_text = data_point['ori_output']
        
        # Extract answer from ori_output if it exists
        try:
            ans = ori_output_text.split('boxed{')[-1].split('}')[0]
        except (IndexError, AttributeError):
            ans = None
        
        # Format base prompt
        try:
            base_prompt = formatInp(
                data_point,
                model=args['model'],
                use_template=args['use_template'],
                tokenizer=tokenizer
            )
        except Exception as e:
            logger.error(f"Error formatting prompt for sample {i}: {e}")
            skipped_samples.append({'index': i, 'reason': 'prompt_formatting_error', 'error': str(e)})
            continue
        
        # Generate dynamic user tag if needed
        current_user_tag = user_tag
        if use_final_result_tag:
            # Extract question from data_point
            question = data_point.get('question', data_point.get('prompt', data_point['problem']))
            current_user_tag = f"""\nThe question is asking {question}. Based on the reasoning so far, the final result to the question is \\boxed{{"""
        
        # Split ori_output into chunks using delimiter
        
        chunks = ori_output_text.split(delimiter)
        if chunks[-1].strip() == '':  # Remove empty last chunk if delimiter ends the text
            chunks = chunks[:-1] 
        if args['till_first_ckpt_right_ans_show']:
            chunks = chunks[:data_point['first_ckpt_show_correct_ans']]
        
        try:
            is_ans_correct = compare_answers(ans, data_point['answer'])
        except Exception:
            is_ans_correct = False
            
        results = {
            'base_prompt': base_prompt,
            'delimiter': delimiter,
            'num_chunks': len(chunks),
            'user_tag': user_tag,
            'current_user_tag': current_user_tag,
            'sample_index': i,
            'is_ans_correct': is_ans_correct,
            'extracted_ans': ans,
            'checkpoint_analysis': {},
            **data_point  # Merge data_point dictionary into results
        }
        
        # Analyze each checkpoint (each checkpoint includes all chunks up to that point)
        sample_oom = False
        first_checkpoint_distribution = None  # Store first checkpoint distribution for KL divergence
        answer_change_tracking = {  # Track answer changes separately
            'sample_index': i,
            'first_checkpoint_answer': None,
            'checkpoint_answer_changes': {}
        }
        stage1_text_tracking = {  # Track stage1 generated text separately
            'sample_index': i,
            'stage1_texts': {}
        }
        if args['max_checkpoint_idx'] is not None and args['max_checkpoint_idx'] < len(chunks)+1:
            max_checkpoint_idx = args['max_checkpoint_idx']
        else:
            max_checkpoint_idx = len(chunks)+1

        if 'checkpoint_analysis' in data_point:
            max_checkpoint_idx = len(data_point['checkpoint_analysis'])
        for checkpoint_idx in range(max_checkpoint_idx):
            try:
                # Combine all chunks up to this checkpoint
                # Apply random number replacement if enabled
                if random_number_replacement:
                    random_seed = args.get('random_seed', 0)
                    if checkpoint_idx==0:
                        chunks_up_to_checkpoint = ''
                        chunk_perturbed = ''
                        is_perturbed = False
                        perturbed_chunks = []

                    elif perturb_forward_all:
                        # New mode: perturb current chunk plus all chunks after it till the end
                        chunks_up_to_checkpoint, is_perturbed, perturbed_chunks = perturb_chunks_range(chunks, checkpoint_idx, len(chunks)+1, random_seed, args['replace_number_mode'],delimiter, replace_all_numbers, rephrase_with_gpt, model=model, tokenizer=tokenizer)
                    elif args['perturb_rand_context_step']:
                        if checkpoint_idx==1: #no previous chunk to perturb
                            chunks_up_to_checkpoint = chunks[0]
                            is_perturbed = False
                            perturbed_chunks = []
                        else:#this ckpt is not included, this is the current chunk
                            end_idx = random.randint(2,checkpoint_idx)
                            start_idx = random.randint(1,end_idx-1)
                            chunks_up_to_checkpoint, is_perturbed, perturbed_chunks = perturb_chunks_range(chunks, start_idx, end_idx, random_seed, args['replace_number_mode'],delimiter, replace_all_numbers, rephrase_with_gpt, model=model, tokenizer=tokenizer)
                        chunks_up_to_checkpoint += '. '+chunks[checkpoint_idx-1]
                    elif args['perturb_forward_before_first_ckpt_right_ans_show']:
                        if checkpoint_idx>=data_point['first_ckpt_show_correct_ans']:
                            break
                        chunks_up_to_checkpoint, is_perturbed, perturbed_chunks = perturb_chunks_range(chunks, checkpoint_idx, data_point['first_ckpt_show_correct_ans'], random_seed, args['replace_number_mode'],delimiter, replace_all_numbers, rephrase_with_gpt, model=model, tokenizer=tokenizer)
                    elif args['perturb_backward_before_first_ckpt_right_ans_show']:
                        if checkpoint_idx<=data_point['first_ckpt_show_correct_ans']:
                            break
                        chunks_up_to_checkpoint, is_perturbed, perturbed_chunks = perturb_chunks_range(chunks, 1, checkpoint_idx, random_seed, args['replace_number_mode'],delimiter, replace_all_numbers, rephrase_with_gpt, model=model, tokenizer=tokenizer)
                        chunks_up_to_checkpoint += '. '+delimiter.join(chunks[checkpoint_idx:data_point['first_ckpt_show_correct_ans']-1])
                    elif args['reverse_perturb']:#perturb the previous chunk
                        if checkpoint_idx==1: #no previous chunk to perturb
                            chunks_up_to_checkpoint = chunks[0]
                            is_perturbed = False
                            perturbed_chunks = []
                        else:#this ckpt is not included, this is the current chunk
                            chunks_up_to_checkpoint, is_perturbed, perturbed_chunks = perturb_chunks_range(chunks, 1, checkpoint_idx, random_seed, args['replace_number_mode'],delimiter, replace_all_numbers, rephrase_with_gpt, model=model, tokenizer=tokenizer)
                        chunks_up_to_checkpoint += '. '+chunks[checkpoint_idx-1]
                    elif args['reverse_perturb_all']:#perturb all the previous chunks plus the current chunk
                        chunks_up_to_checkpoint, is_perturbed, perturbed_chunks = perturb_chunks_range(chunks, 1, checkpoint_idx+1, random_seed, args['replace_number_mode'],delimiter, replace_all_numbers, rephrase_with_gpt, model=model, tokenizer=tokenizer)
                    else:
                        # Original mode: perturb only the previous chunk
                        if args['replace_number_mode'] == 'replace_all_tokens_ellipsis':
                            chunk_perturbed, is_perturbed = replace_all_tokens_with_ellipsis(
                                chunks[checkpoint_idx-1], tokenizer=tokenizer
                            )
                        else:
                            chunk_perturbed, is_perturbed = randomly_replace_numbers(
                                chunks[checkpoint_idx-1], random_seed, args['replace_number_mode'],
                                replace_all_numbers, rephrase_with_gpt, model=model, tokenizer=tokenizer
                            )
                        if remove_boxed_answers_flag:
                            chunk_perturbed = remove_boxed_answers(chunk_perturbed)
                        chunks_up_to_checkpoint = delimiter.join(chunks[:checkpoint_idx-1]) + chunk_perturbed 
                        logger.info(f"Chunk perturbation applied: {is_perturbed}")
                        if is_perturbed:
                            logger.info(f"Perturbed chunk: {chunk_perturbed}")
                            logger.info(f"Original chunk: {chunks[checkpoint_idx-1]}")
                    checkpoint_prompt = base_prompt + chunks_up_to_checkpoint
                else:
                    chunks_up_to_checkpoint = delimiter.join(chunks[:checkpoint_idx ])
                    # Create prompt: base_prompt + chunks_up_to_checkpoint + user_tag
                    checkpoint_prompt = base_prompt + chunks_up_to_checkpoint + current_user_tag
                   
                 
                # If random number replacement is enabled, do two-stage generation
                if random_number_replacement:
                    
                    checkpoint_prompt_w_user_tag = checkpoint_prompt + current_user_tag

                    analysis_w_user_tag = generate_and_analyze_checkpoint(
                        model, tokenizer, checkpoint_prompt_w_user_tag, ans, checkpoint_idx, 
                        top_k, store_logits, store_distribution
                    )
                    
                    # Check if analysis resulted in error
                    if 'error' in analysis_w_user_tag:
                        logger.error(f"Error in analysis for checkpoint {checkpoint_idx + 1}: {analysis_w_user_tag.get('error_details', 'Unknown error')}")
                        # Store the error analysis and continue with next checkpoint
                        results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis_w_user_tag
                        continue
                    
                    ans_w_user_tag = analysis_w_user_tag['extracted_generated_answer']
                    ans_w_user_tag_prob = analysis_w_user_tag['extracted_generated_answer_probability']

                    if not perturb_forward_all and not args['skip_stage2']:
                        print(f"Two-stage generation for checkpoint {checkpoint_idx + 1}")
                        logger.info(f"Two-stage generation for checkpoint {checkpoint_idx + 1}")
                        
                        # Generate full output using vLLM for faster inference
                        logger.info(f"Using vLLM for stage 1 generation")
                        # Generate stage 1 text using vLLM
                        generated_text_stage1 = generate_with_vllm(vllm_model, checkpoint_prompt, max_new_tokens_stage1)
                        
                        print(f"Stage 1 generated text: {generated_text_stage1}")
                        logger.info(f"Stage 1 generated text: {generated_text_stage1}")
                        
                        # Stage 2: Use generated text + user_tag to get final answer
                        stage2_prompt = checkpoint_prompt + generated_text_stage1 + current_user_tag
                        print(f"Stage 2 prompt: {stage2_prompt}")
                        logger.info(f"Stage 2 prompt: {stage2_prompt}")
                        
                        # Analyze stage 2 using helper function
                        analysis = generate_and_analyze_checkpoint(
                            model, tokenizer, stage2_prompt, ans, checkpoint_idx, 
                            top_k, store_logits, store_distribution
                        )
                        
                        # Check if analysis resulted in error
                        if 'error' in analysis:
                            logger.error(f"Error in stage 2 analysis for checkpoint {checkpoint_idx + 1}: {analysis.get('error_details', 'Unknown error')}")
                            # Store the error analysis and continue with next checkpoint
                            results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis
                            continue
                        
                        # Store stage 1 text for two-stage generation
                        stage1_text_tracking['stage1_texts'][f'checkpoint_{checkpoint_idx + 1}'] = generated_text_stage1
                        analysis['stage1_text_after_perturb'] = generated_text_stage1
                    else:
                        # For perturb_forward_all mode, use the analysis with user tag directly
                        analysis = analysis_w_user_tag
                        # No stage 1 text for forward perturbation mode
                        stage1_text_tracking['stage1_texts'][f'checkpoint_{checkpoint_idx + 1}'] = ""
                        
                    # Add perturbation information and mark as perturbed
                    analysis['input_prompt'] = checkpoint_prompt_w_user_tag
                    analysis['perturbed'] = is_perturbed
                    analysis['perturb_forward_all'] = perturb_forward_all
                    if perturb_forward_all or args['perturb_forward_before_first_ckpt_right_ans_show'] or args['reverse_perturb'] or args['reverse_perturb_all']:
                        
                        # For forward perturbation, save information about which chunks were perturbed
                        #analysis['text_up_to_checkpoint_initial'] = chunks[checkpoint_idx-1] if checkpoint_idx > 0 else ''
                        #analysis['perturbed_chunks_range'] = f"chunks[{checkpoint_idx}:] (current and all after)"
                        if len(perturbed_chunks)>0:
                            analysis['perturbed_chunks'] = delimiter.join(perturbed_chunks)
                            analysis['text_up_to_checkpoint_perturbed'] = perturbed_chunks[-1]
                        else:
                            analysis['perturbed_chunks'] = ''
                            analysis['text_up_to_checkpoint_perturbed'] = ''
                    else:
                        # Original mode: save the specific chunk that was perturbed
                        analysis['perturbed_chunks_range'] = f"chunks[{checkpoint_idx-1}] (previous chunk only)"
                        analysis['text_up_to_checkpoint_perturbed'] = chunk_perturbed

                    analysis['text_up_to_checkpoint'] = chunks[checkpoint_idx-1] if checkpoint_idx > 0 else ''
                    
                    # Remove the avg_distribution tensor to avoid JSON serialization issues
                    if 'avg_distribution' in analysis:
                        del analysis['avg_distribution']
                    
                    # Track answer changes separately
                    current_answer = analysis['extracted_generated_answer']
                    if checkpoint_idx == 0:
                        answer_change_tracking['first_checkpoint_answer'] = current_answer
                        answer_change_tracking['checkpoint_answer_changes'][f'checkpoint_{checkpoint_idx + 1}'] = {
                            'answer': current_answer,
                            'ans_not_change_after_replace': True  # First checkpoint is baseline
                        }
                    else:
                        first_answer = answer_change_tracking['first_checkpoint_answer']
                        ans_not_changed = (current_answer == first_answer)
                        answer_change_tracking['checkpoint_answer_changes'][f'checkpoint_{checkpoint_idx + 1}'] = {
                            'answer': current_answer,
                            'ans_not_change_after_replace': ans_not_changed
                        }
                    # Store initial early exit answer information if available
                    if 'checkpoint_analysis' not in data_point:
                        try:
                            analysis['prob_initial_early_exit_answer'] = data_point[f'checkpoint_{checkpoint_idx + 1}']['extracted_generated_answer_probability']
                            analysis['initial_early_exit_answer'] = data_point[f'checkpoint_{checkpoint_idx + 1}']['extracted_generated_answer']
                        except KeyError:
                            pass
                    else:
                        analysis['prob_initial_early_exit_answer'] = data_point['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}']['extracted_generated_answer_probability']
                        analysis['initial_early_exit_answer'] = data_point['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}']['extracted_generated_answer']
                    
                    analysis['prob_perturbed_early_exit_answer'] = ans_w_user_tag_prob
                    analysis['perturbed_early_exit_answer'] = ans_w_user_tag
                    results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis

                else:
                    # Original single-stage generation logic
                    
                    if args['continue_after_early_exit']:
                        checkpoint_prompt += f"{data_point['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}']['extracted_generated_answer']}}}"
                    logger.info(f"Checkpoint prompt: {checkpoint_prompt}")
                    logger.info(f"Analyzing checkpoint {checkpoint_idx + 1}/{len(chunks)}")
                    
                    # Use helper function for generation with OOM handling
                    try:
                        analysis = generate_and_analyze_checkpoint(
                            model, tokenizer, checkpoint_prompt, ans, checkpoint_idx,
                            top_k, store_logits, store_distribution
                        )
                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"OOM error in generate_and_analyze_checkpoint for checkpoint {checkpoint_idx + 1}: {e}")
                        # Aggressive GPU cleanup
                        clear_gpu_cache()
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                            torch.cuda.reset_peak_memory_stats()
                        
                        # Create OOM error analysis
                        analysis = {
                            'checkpoint_idx': checkpoint_idx,
                            'text_up_to_checkpoint': chunks[checkpoint_idx-1] if checkpoint_idx>0 else '',
                            'input_length': 0,
                            'error': 'generation_oom',
                            'error_details': str(e),
                            'top_predictions': [],
                            'ans_probability': None,
                            'ans_token_id': None,
                            'full_generated_text': '',
                            'generation_sequence': [],
                            'extracted_generated_answer': None
                        }
                        results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error in generate_and_analyze_checkpoint for checkpoint {checkpoint_idx + 1}: {e}")
                        # Create general error analysis
                        analysis = {
                            'checkpoint_idx': checkpoint_idx,
                            'text_up_to_checkpoint': chunks[checkpoint_idx-1] if checkpoint_idx>0 else '',
                            'input_length': 0,
                            'error': 'generation_error',
                            'error_details': str(e),
                            'top_predictions': [],
                            'ans_probability': None,
                            'ans_token_id': None,
                            'full_generated_text': '',
                            'generation_sequence': [],
                            'extracted_generated_answer': None
                        }
                        results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis
                        continue
                    
                    # Check if analysis resulted in error
                    
                    if 'error' in analysis or 'extracted_generated_answer' not in analysis:
                        logger.error(f"Error in analysis for checkpoint {checkpoint_idx + 1}: {analysis.get('error_details', 'Unknown error')}")
                        # Store the error analysis and continue with next checkpoint
                        results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis
                        analysis['error'] = 'unknown_error'
                        continue
                    
                    
                    analysis['text_up_to_checkpoint'] = chunks[checkpoint_idx-1] if checkpoint_idx > 0 else ''
                    
                    # Remove the avg_distribution tensor to avoid JSON serialization issues
                    if 'avg_distribution' in analysis:
                        del analysis['avg_distribution']
                    
                    # Track answer changes separately - only if we have a valid extracted answer
                    current_answer = analysis.get('extracted_generated_answer', None)
                    if current_answer is None:
                        logger.warning(f"No extracted answer for checkpoint {checkpoint_idx + 1}, skipping answer tracking")
                        results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis
                        continue
                    if checkpoint_idx == 0:
                        answer_change_tracking['first_checkpoint_answer'] = current_answer
                        answer_change_tracking['checkpoint_answer_changes'][f'checkpoint_{checkpoint_idx + 1}'] = {
                            'answer': current_answer,
                            'ans_not_change_after_replace': True  # First checkpoint is baseline
                        }
                    else:
                        first_answer = answer_change_tracking['first_checkpoint_answer']
                        ans_not_changed = (current_answer == first_answer)
                        answer_change_tracking['checkpoint_answer_changes'][f'checkpoint_{checkpoint_idx + 1}'] = {
                            'answer': current_answer,
                            'ans_not_change_after_replace': ans_not_changed
                        }
                    
                    results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis
                    
                    # Log generation results for this checkpoint
                    if 'top_predictions' in analysis and analysis['top_predictions']:
                        top_pred = analysis['top_predictions'][0]
                        print(top_pred)
                        logger.info(f"Checkpoint {checkpoint_idx + 1}: Top prediction '{top_pred[0]['token_text']}' (prob: {top_pred[0]['probability']:.4f})")
      
                # Clear GPU cache after each checkpoint to prevent memory buildup
                clear_gpu_cache()
                
                # Log memory every 50 checkpoints to reduce overhead
                if checkpoint_idx % 50 == 0:
                    log_gpu_memory()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU OOM error for sample {i} checkpoint {checkpoint_idx}: {e}")
                sample_oom = True
                skipped_samples.append({
                    'index': i, 
                    'checkpoint': checkpoint_idx,
                    'reason': 'gpu_oom',
                    'error': str(e)
                })
                
                # Aggressive GPU cleanup
                clear_gpu_cache()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
                
                # Log memory status after cleanup
                if torch.cuda.is_available():
                    logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved")
                
                # Add error entry to results
                analysis = {
                    'checkpoint_idx': checkpoint_idx,
                    'text_up_to_checkpoint': chunks_up_to_checkpoint if 'chunks_up_to_checkpoint' in locals() else '',
                    'input_length': 0,
                    'error': 'gpu_oom',
                    'error_details': str(e),
                    'top_predictions': [],
                    'ans_probability': None,
                    'ans_token_id': None,
                    'full_generated_text': '',
                    'generation_sequence': [],
                    'extracted_generated_answer': None
                }
                results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis
                break  # Skip remaining checkpoints for this sample
            except Exception as e:
                logger.error(f"Error in generate_and_analyze_checkpoint for checkpoint {checkpoint_idx + 1}: {e}")
                analysis = {
                    'checkpoint_idx': checkpoint_idx,
                    'text_up_to_checkpoint': chunks[checkpoint_idx-1] if checkpoint_idx>0 else '',
                    'input_length': 0,
                    'error': 'generation_error',
                    'error_details': str(e),
                    'top_predictions': [],
                    'ans_probability': None,
                    'ans_token_id': None,
                    'full_generated_text': '',
                    'generation_sequence': [],
                    'extracted_generated_answer': None
                }
                results['checkpoint_analysis'][f'checkpoint_{checkpoint_idx + 1}'] = analysis
                continue
        # Save results with proper formatting
        
        with open(args['output_file_name'], 'a', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
            f.write('\n')
        logger.info(f"Saved results for sample {i+1}/{len(eval_data)}")

        
        logger.debug(f'Processed sample {i+1}/{len(eval_data)}')
        processed_samples += 1
        
        # Save progress every 10 samples
        if processed_samples % 10 == 0:
            progress_data = {
                'processed_samples': processed_samples,
                'total_samples': len(eval_data),
                'percentage': (processed_samples / len(eval_data)) * 100,
                'timestamp': time.time(),
                'elapsed_time': time.time() - start_time,
                'skipped_count': len(skipped_samples)
            }
            try:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, indent=2)
                logger.info(f"Progress: {processed_samples}/{len(eval_data)} ({progress_data['percentage']:.1f}%)")
            except Exception as e:
                logger.warning(f"Failed to save progress: {e}")
        
        # Clear GPU cache after each sample
        clear_gpu_cache()
    
    end_time = time.time()
    logger.info(f'Checkpoint analysis completed in {end_time - start_time:.2f} seconds')
    
    # Save final progress
    final_progress = {
        'processed_samples': processed_samples,
        'total_samples': len(eval_data),
        'percentage': 100.0,
        'timestamp': end_time,
        'elapsed_time': end_time - start_time,
        'skipped_count': len(skipped_samples),
        'status': 'completed'
    }
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(final_progress, f, indent=2)
        logger.info(f"Final progress saved to {progress_file}")
    except Exception as e:
        logger.warning(f"Failed to save final progress: {e}")
    
    # Log summary of skipped samples
    if skipped_samples:
        logger.warning(f"Skipped {len(skipped_samples)} samples due to errors:")
        for skip in skipped_samples:
            logger.warning(f"  Sample {skip['index']}: {skip['reason']} - {skip.get('error', '')}")
    
    # Save skipped samples summary
    skipped_file = args['output_file_name'].replace('.jsonl', '_skipped_samples.json')
    with open(skipped_file, 'w') as f:
        json.dump(skipped_samples, f, indent=2)
    logger.info(f"Skipped samples summary saved to: {skipped_file}")

    # Log answer change tracking file if random replacement was enabled
    if args.get('random_number_replacement', False):
        answer_change_file = args['output_file_name'].replace('.jsonl', '_answer_changes.jsonl')
        if os.path.exists(answer_change_file):
            logger.info(f"Answer change tracking saved to: {answer_change_file}")
        else:
            logger.warning(f"Answer change file was not created: {answer_change_file}")
        
        stage1_text_file = args['output_file_name'].replace('.jsonl', '_stage1_texts.jsonl')
        if os.path.exists(stage1_text_file):
            logger.info(f"Stage1 generated texts saved to: {stage1_text_file}")
        else:
            logger.warning(f"Stage1 text file was not created: {stage1_text_file}")


def main():
    """Main function to run the checkpoint analysis script."""
    parser = argparse.ArgumentParser(description="Checkpoint-based token analysis script")
    
    # Model configuration
    parser.add_argument("--model", default='llama3', type=str, help="Model type (llama, llama3, qwen, olmo)")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size (7b, 8b, 13b, 14b, 32b)")
    parser.add_argument("--tensor_parallel_size", default=2, type=int, help="Number of GPUs for tensor parallelism (default: 2 for dual GPU)")
    parser.add_argument("--gpu_memory_utilization", default=0.6, type=float, help="GPU memory utilization ratio (default: 0.6)")
    parser.add_argument("--vllm_single_gpu", action='store_true', help="Use only one GPU for vLLM (useful when transformers model uses other GPU)")
    
    # Data configuration
    parser.add_argument("--input", default='data/medcq.json', type=str, help="Input file path")
    parser.add_argument("--output_file_name", default='output/checkpoint_analysis_results.jsonl', type=str, help="Output file path")
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=10, type=int, help='Right index for data slicing')
    
    # Analysis configuration
    parser.add_argument("--delimiter", default='.', type=str, help="String delimiter to split ori_output on")
    parser.add_argument("--user_tag", default='', type=str, help="User tag to append after checkpoint text")
    parser.add_argument("--top_k", default=10, type=int, help="Number of top predictions to return")
    parser.add_argument("--store_logits", action='store_true', default=True, help="Store full logits tensor")
    parser.add_argument("--no_store_logits", dest='store_logits', action='store_false', help="Disable storing full logits tensor")
    parser.add_argument("--store_distribution", action='store_true', default=True, help="Store full distribution tensor")
    parser.add_argument("--no_store_distribution", dest='store_distribution', action='store_false', help="Disable storing full distribution tensor")
    parser.add_argument("--random_number_replacement", action='store_true', help="Randomly replace one number in each checkpoint text")
    parser.add_argument("--perturb_forward_all", action='store_true', help="Perturb current chunk plus all chunks after it till the end (requires --random_number_replacement)")
    parser.add_argument("--replace_all_numbers", action='store_true', help="Replace all numbers instead of just one (requires --random_number_replacement)")
    parser.add_argument("--rephrase_with_gpt", action='store_true', help="Rephrase text using GPT-4o before number replacement (requires --random_number_replacement)")
    parser.add_argument("--remove_boxed_answers", action='store_true', help="Remove \\boxed{} patterns from perturbed chunks")
    parser.add_argument("--use_final_result_tag", action='store_true', help="Use final result user tag format instead of default [CHECKPOINT] tag")
    parser.add_argument("--use_vllm", default=0, type=int,help="Use vLLM for faster stage 1 generation (if available)")
    parser.add_argument('--skip_stage2', default=1, type=int, help="Skip stage 2 generation for the first n checkpoints")
    parser.add_argument('--perturb_forward_before_first_ckpt_right_ans_show', default=0, type=int, help="Perturb forward before first checkpoint right answer show")
    parser.add_argument('--perturb_backward_before_first_ckpt_right_ans_show', default=0, type=int, help="Perturb forward before first checkpoint right answer show all")
    parser.add_argument('--replace_number_mode', default='random', type=str, help="Mode for number replacement: 'random', 'random_add_small', 'random_add_large', 'close_embed', 'far_embed', 'replace_all_tokens_ellipsis', 'gpt4_distort'")
    parser.add_argument("--random_seed", default=0, type=int, help="Random seed for reproducible number replacement (only used with --random_number_replacement)")
    
    # Prompt configuration
    parser.add_argument("--use_template", default=1, type=int, help="Use default prompting template")
    parser.add_argument("--till_first_ckpt_right_ans_show", default=0, type=int, help="Till first checkpoint right answer show")
    parser.add_argument("--continue_after_early_exit", default=0, type=int, help="Continue after early exit")
    parser.add_argument("--reverse_perturb", default=0, type=int, help="Reverse perturb: perturb all the previous chunks")
    parser.add_argument("--reverse_perturb_all", default=0, type=int, help="Reverse perturb: perturb all the previous chunks plus the current chunk")
    parser.add_argument("--max_checkpoint_idx", default=None, type=int, help="Max checkpoint to analyze in case of OOM")
    parser.add_argument("--perturb_rand_context_step", default=0, type=int, help="Perturb random context step")
    args = parser.parse_args()
    params = vars(args)
    
    # Validate argument combinations
    if params['skip_stage2'] == 1:
        assert params['use_vllm'] == 0, "vLLM is not supported for skip_stage2"
    # Log configuration for startup
    logger.info(f'Model: {params["model"]} {params["model_size"]} | Delimiter: "{params["delimiter"]}" | vLLM: {params["use_vllm"]}')
    if params["random_number_replacement"]:
        logger.info(f'Random replacement enabled | Mode: {params["replace_number_mode"]} | Seed: {params["random_seed"]}')
    
    # Log environment information
    if is_runpod_environment():
        logger.info("RunPod environment detected")
    
    # Load model and tokenizer with tensor parallelism
    model, tokenizer = load_model_and_tokenizer_optimized(params['model'], params['model_size'], params['use_vllm'])
    
    # Log memory usage after model loading
    log_gpu_memory()
    
    # Load vLLM model for faster stage 1 generation (optional)
    vllm_model = None
    print(f"VLLM_AVAILABLE: {VLLM_AVAILABLE}")
    print(f"params.get('use_vllm'): {params['use_vllm']}")
    if  params["use_vllm"] and VLLM_AVAILABLE :
      
        # Use single GPU for vLLM if transformers model is already loaded
        vllm_tensor_parallel = 1 if params.get('vllm_single_gpu', False) else params.get('tensor_parallel_size', 2)
        # Use GPU 1 for vLLM when single GPU mode is enabled
        gpu_devices = "1" if params.get('vllm_single_gpu', False) else None
        vllm_model, _ = load_vllm_model_and_tokenizer_optimized(
            params['model'], 
            params['model_size'],
            tensor_parallel_size=vllm_tensor_parallel,
            gpu_memory_utilization=params.get('gpu_memory_utilization', 0.6),
            trust_remote_code=True,
            gpu_devices=gpu_devices
        )
        if vllm_model is not None:
            logger.info("vLLM model loaded successfully for stage 1 generation")
        else:
            logger.warning("Failed to load vLLM model, will use transformers for all generation")

    elif not VLLM_AVAILABLE:
        logger.info("vLLM not available, using transformers for all generation")
    elif not params["use_vllm"]:
        logger.info("vLLM disabled by user, using transformers for all generation")
    # vLLM model is optional - if not available, will use transformers
    # Load and preprocess data efficiently
    logger.info(f"Loading data from {params['input']}")
    test_data = read_row(params['input'])
    
    # Validate and filter data in one pass
    left = max(0, params['left'])
    right = min(len(test_data), params['right'])
    
    if left >= right:
        raise ValueError("Left index must be less than right index")
    
    test_data = test_data[left:right]
    logger.info(f"Processing {len(test_data)} samples (indices {left}-{right})")
    
    # Run checkpoint analysis
    analyze_checkpoints_for_dataset(model, tokenizer, test_data, params, vllm_model)


if __name__ == "__main__":
    main() 
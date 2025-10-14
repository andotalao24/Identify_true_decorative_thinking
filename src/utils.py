import json
import numpy as np
import pickle
import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig,AutoModel
from typing import Dict, List, Tuple, Any, Optional
import logging
from vllm import LLM
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_type: str, model_size: str, use_vllm: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the specified model and tokenizer.
    
    Args:
        model_type: Type of model ('llama', 'llama3', 'qwen', 'olmo', 'deepseek-qwen')
        model_size: Size of model ('7b', '8b', '13b', '14b', '32b')
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        model_path = get_model_path(model_type, model_size)
        
        # Setup workspace directories for RunPod
        workspace_cache_dir, workspace_models_dir = setup_workspace_directories()
        cache_kwargs = get_model_cache_kwargs()
        
        logger.info(f"Loading model: {model_path}")
        if is_runpod_environment():
            logger.info(f"RunPod detected - downloading to workspace: {workspace_models_dir}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            **cache_kwargs
        )
        
        # Load model with optimized device mapping for multi-GPU
        if torch.cuda.device_count() > 1 and use_vllm:
            # For multi-GPU, prefer GPU 0 for transformers model
            device_map = "cuda:0"  # Force GPU 0
            logger.info("Using GPU 0 for transformers model")
        else:
            device_map = "auto"
            logger.info("Using auto device mapping for single GPU")
        

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




def get_model_path(model_type: str, model_size: str) -> str:
    """
    Get the model path for the specified model type and size.
    
    Args:
        model_type: Type of model ('llama', 'llama3', 'qwen', 'olmo', 'deepseek-qwen')
        model_size: Size of model ('7b', '8b', '13b', '14b', '32b')
        
    Returns:
        Model path string
    """
    if model_type == 'llama':
        if model_size == '7b':
            return "NousResearch/Llama-2-7b-chat-hf"
        elif model_size == '13b':
            return "NousResearch/Llama-2-13b-chat-hf"
        else:
            raise ValueError(f"Unsupported model size for llama: {model_size}")
        
    elif model_type == 'llama3':
        if model_size == '8b':
            return "unsloth/Meta-Llama-3.1-8B-Instruct"
        elif model_size == '13b':
            return "unsloth/Meta-Llama-3.1-13B-Instruct"
        else:
            raise ValueError(f"Unsupported model size for llama3: {model_size}")
            
    elif model_type == 'qwen':
        if model_size == '7b':
            return "Qwen/Qwen2.5-Math-7B"
        elif model_size == '8b':
            return "Qwen/Qwen3-8B"
        elif model_size == '14b':
            return "Qwen/Qwen2.5-14B"
        elif model_size == '32b':
            return "Qwen/Qwen2.5-32B"
        else:
            raise ValueError(f"Unsupported model size for qwen: {model_size}")
            
    elif model_type == 'olmo':
        if model_size == '7b':
            return "allenai/OLMo-2-1124-7B-Instruct"
        elif model_size == '13b':
            return "allenai/OLMo-2-1124-13B-Instruct"
        elif model_size == '32b':
            return "allenai/OLMo-2-0325-32B-Instruct"
        else:
            raise ValueError(f"Unsupported model size for olmo: {model_size}")
            
    elif model_type == 'deepseek-qwen':
        if model_size == '8b':
            return "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        elif model_size == '7b':
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        elif model_size == '14b':
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        elif model_size == '32b':
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        elif model_size == '1.5b':
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        else:
            raise ValueError(f"Unsupported model size for deepseek-qwen: {model_size}")
    elif model_type == 'deepseek-llama':
        if model_size == '8b':
            return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        elif model_size == '14b':
            return "deepseek-ai/DeepSeek-R1-Distill-Llama-14B"
        else:
            raise ValueError(f"Unsupported model size for deepseek-llama: {model_size}")
    elif model_type == 'nemotron':
        if model_size == '9b':
            return "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
        elif model_size == '1.5b':
            return "nvidia/OpenReasoning-Nemotron-1.5B"
        else:
            raise ValueError(f"Unsupported model size for nemotron: {model_size}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def setup_vllm_environment(workspace_cache_dir: str, workspace_models_dir: str):
    """Setup environment variables for vLLM to use workspace directories."""
    if is_runpod_environment():
        # Set all environment variables that vLLM and HuggingFace might use
        env_vars = {
            'VLLM_CACHE_DIR': workspace_models_dir,
            'VLLM_DOWNLOAD_DIR': workspace_models_dir,
            'HF_HOME': workspace_cache_dir,
            'TRANSFORMERS_CACHE': os.path.join(workspace_cache_dir, 'transformers'),
            'HF_HUB_CACHE': os.path.join(workspace_cache_dir, 'hub'),
            'HF_DATASETS_CACHE': os.path.join(workspace_cache_dir, 'datasets'),
            'HF_MODELS_CACHE': workspace_models_dir,
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            logger.info(f"Set {var} = {value}")


def load_vllm_model_and_tokenizer(
    model_type: str, 
    model_size: str, 
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    gpu_devices: Optional[str] = None
) :
    """
    Load the specified model and tokenizer using vLLM.
    
    Args:
        model_type: Type of model ('llama', 'llama3', 'qwen', 'olmo', 'deepseek-qwen')
        model_size: Size of model ('7b', '8b', '13b', '14b', '32b')
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization ratio
        max_model_len: Maximum model length
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (vllm_model, tokenizer)
    """
    try:
        model_path = get_model_path(model_type, model_size)
        
        # Setup workspace directories
        workspace_cache_dir, workspace_models_dir = setup_workspace_directories()
        
        # Setup vLLM environment variables
        setup_vllm_environment(workspace_cache_dir, workspace_models_dir)
        
        # Load tokenizer first to get model configuration
        cache_kwargs = get_model_cache_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **cache_kwargs
        )
        
        # Get model configuration to determine max_model_len
        config = AutoConfig.from_pretrained(
            model_path, 
            trust_remote_code=trust_remote_code,
            **cache_kwargs
        )
        
        # Query the model's actual maximum length from its configuration
        if max_model_len is None:
            # Try different possible attribute names for max length
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
                # Fallback to model-specific defaults if config doesn't have the info
                logger.warning(f"Could not determine max length from config for {model_type} {model_size}, using defaults")
                if model_type == 'olmo':
                    if model_size == '32b':
                        max_model_len = 32768
                    else:
                        max_model_len = 8192
                elif model_type == 'llama3':
                    max_model_len = 8192
                elif model_type == 'llama':
                    max_model_len = 4096
                elif model_type == 'qwen':
                    max_model_len = 32768
                elif model_type == 'deepseek-qwen':
                    max_model_len = 32768
                else:
                    max_model_len = 4096
        
        logger.info(f"Loading vLLM model: {model_path}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"Max model length (from config): {max_model_len}")
        
        # Load vLLM model with aggressive workspace forcing
        vllm_kwargs = {
            'model': model_path,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_model_len': max_model_len,
            'trust_remote_code': trust_remote_code,
            'dtype': "auto",  # Automatically select best dtype
            'enforce_eager': False,  # Use CUDA graphs for better performance
        }
        
        # Environment variables are already set by setup_vllm_environment
        # Just ensure we don't pass invalid parameters to vLLM
        vllm_kwargs.pop('download_dir', None)  # Remove if exists to avoid errors
        
        # Temporarily set CUDA_VISIBLE_DEVICES to control GPU usage for vLLM
        original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if gpu_devices:
            logger.info(f"vLLM will use GPU devices: {gpu_devices}")
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
        
        vllm_model = None
        try:
            vllm_model = LLM(**vllm_kwargs)
        finally:
            # Restore original CUDA_VISIBLE_DEVICES
            if original_cuda_visible_devices is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
        
        logger.info(f"Successfully loaded vLLM model: {model_path}")
        return vllm_model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load vLLM model {model_type} {model_size}: {e}")
        raise


def get_model_max_length(model: AutoModelForCausalLM) -> int:
    """
    Get the maximum context length from the model's configuration.
    
    Args:
        model: The loaded model
        
    Returns:
        Maximum context length
    """
    # Try different possible attribute names for max length
    for attr_name in ['max_position_embeddings', 'max_sequence_length', 'context_length', 'seq_length', 'n_positions']:
        if hasattr(model.config, attr_name):
            return getattr(model.config, attr_name)
    
    logger.warning("Could not determine max length from model config")
    return 4096  # Default fallback


def is_runpod_environment() -> bool:
    """Check if we're running in a RunPod environment."""
    return os.path.exists('/workspace') or os.environ.get('RUNPOD_POD_ID') is not None

def setup_environment_variables():
    """Setup all environment variables to force downloads to workspace."""
    if is_runpod_environment():
        # Core HuggingFace cache directories
        os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/transformers'
        os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
        
        # Additional cache directories that might be used
        os.environ['HF_HUB_CACHE'] = '/workspace/.cache/huggingface/hub'
        os.environ['HF_MODELS_CACHE'] = '/workspace/models'
        
        # vLLM specific environment variables
        os.environ['VLLM_CACHE_DIR'] = '/workspace/models'
        os.environ['VLLM_DOWNLOAD_DIR'] = '/workspace/models'
        
        # Set HuggingFace to not use default cache
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        # Force HuggingFace to use our cache
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['HF_HUB_LOCAL_FILES_ONLY'] = '0'
        
        print("Environment variables set for RunPod workspace")

# Initialize environment variables only once at module import
if not hasattr(setup_environment_variables, '_called'):
    setup_environment_variables()
    setup_environment_variables._called = True

def setup_workspace_directories() -> Tuple[str, str]:
    """Setup workspace directories for model caching."""
    if is_runpod_environment():
        workspace_cache_dir = '/workspace/.cache/huggingface'
        workspace_models_dir = '/workspace/models'
    else:
        workspace_cache_dir = os.path.expanduser('~/.cache/huggingface')
        workspace_models_dir = os.path.join(workspace_cache_dir, 'models')
    
    os.makedirs(workspace_cache_dir, exist_ok=True)
    os.makedirs(workspace_models_dir, exist_ok=True)
    
    return workspace_cache_dir, workspace_models_dir

def clear_wrong_cache_locations():
    """Clear cache files from wrong locations (like root) in RunPod."""
    if not is_runpod_environment():
        return
    
    import shutil
    
    wrong_locations = [
        '/root/.cache/huggingface',
        '/root/.huggingface',
        '/root/.cache/transformers',
    ]
    
    for location in wrong_locations:
        if os.path.exists(location):
            try:
                shutil.rmtree(location)
            except Exception as e:
                logger.warning(f"Could not clear {location}: {e}")

def setup_vllm_environment(workspace_cache_dir: str, workspace_models_dir: str):
    """Setup environment variables for vLLM to use workspace directories."""
    if is_runpod_environment():
        # Set all environment variables that vLLM and HuggingFace might use
        env_vars = {
            'VLLM_CACHE_DIR': workspace_models_dir,
            'VLLM_DOWNLOAD_DIR': workspace_models_dir,
            'HF_HOME': workspace_cache_dir,
            'TRANSFORMERS_CACHE': os.path.join(workspace_cache_dir, 'transformers'),
            'HF_HUB_CACHE': os.path.join(workspace_cache_dir, 'hub'),
            'HF_DATASETS_CACHE': os.path.join(workspace_cache_dir, 'datasets'),
            'HF_MODELS_CACHE': workspace_models_dir,
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value

def get_model_cache_kwargs() -> dict:
    """Get keyword arguments for model loading with proper cache directory."""
    workspace_cache_dir, workspace_models_dir = setup_workspace_directories()
    
    # Clear any wrong cache locations first
    clear_wrong_cache_locations()
    
    kwargs = {
        'cache_dir': workspace_models_dir,
        'local_files_only': False,
        'force_download': False,  # Don't force download if files exist
        'resume_download': True,  # Resume partial downloads
        'proxies': None,  # Ensure no proxy interference
    }
    
    # Additional parameters to ensure workspace usage
    if is_runpod_environment():
        kwargs.update({
            'use_auth_token': None,  # Don't use auth token that might affect cache
            'mirror': None,  # Don't use mirrors that might bypass our cache
        })
    
    return kwargs

# Global tokenizer cache to avoid reloading
_tokenizer_cache = {}

def read_row(file):
    #return list of dictionaries
    ret=[]
    try:
        with open(file,'r', encoding="UTF-8") as f:
            for row in f.readlines():
                d=json.loads(row)
                ret.append(d)
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        with open(file,'r') as f:
            ret=json.load(f)
    return ret


def store_row(file,ret):
    with open(file,'w') as f:
        for row in ret:
            json.dump(row,f)
            f.write('\n')

def read_pkl(pickle_file_path):
    try:
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"File {pickle_file_path} not found.")
    except pickle.UnpicklingError:
        print("Error unpickling the file.")

def _get_tokenizer(model_name):
    """Get tokenizer from cache or load it"""
    if model_name not in _tokenizer_cache:
        try:
            cache_kwargs = get_model_cache_kwargs()
            _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                **cache_kwargs
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {model_name}: {e}")
            return None
    return _tokenizer_cache[model_name]

def formatInp(d, model='llama2', use_template=True, tokenizer=None,use_system_prompt=False):
    '''
    Format input data using Hugging Face's built-in chat templates
    :param d: input data point, dictionary
    :param model: model name/type for tokenizer (used only if tokenizer is None)
    :param use_template: whether to apply chat formatting
    :param tokenizer: pre-loaded tokenizer (if provided, model parameter is ignored)
    :return: a string that is the input to the model
    '''
   
    sys_prompt="You are a helpful assistant, and you are not easily persuaded, and you know what is right and wrong"
    
    # Extract the main content from the data dictionary
    content_keys = ['question', 'problem', 'input', 'instruction', 'text', 'content']
    content = None
    
    for key in content_keys:
        if key in d:
            content = d[key]
            break
    
    if content is None:
        # If no standard key found, try to get the first string value
        for key, value in d.items():
            if isinstance(value, str):
                content = value
                break
    
    if content is None:
        raise ValueError("Could not find content in data dictionary")
    
    if not use_template:
        return content
    
    # Use HF's built-in chat templates
    try:
        # If tokenizer is provided, use it directly
        if tokenizer is not None:
            if hasattr(tokenizer, 'apply_chat_template'):
                # Create messages format for chat template with system prompt
                if use_system_prompt:
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": content}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": content}
                    ]
                
                # Apply the chat template
                formatted = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted
            else:
                # Fallback to manual formatting if no chat template available
                return _manual_chat_format(content, model)
        
        # Otherwise, load tokenizer based on model name
        model_mapping = {
            'llama': 'meta-llama/Llama-2-7b-chat-hf',
            'llama2': 'meta-llama/Llama-2-7b-chat-hf', 
            'llama3': 'meta-llama/Llama-3-8B-Instruct',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
            'vicuna': 'lmsys/vicuna-7b-v1.5',
            'qwen': 'Qwen/Qwen2-7B-Instruct',
            'qwen2': 'Qwen/Qwen2-7B-Instruct',
            'olmo': 'allenai/OLMo-7B-Instruct',
            'olmo-instruct': 'allenai/OLMo-7B-Instruct'
        }
        
        model_name = model_mapping.get(model.lower(), model)
        
        # Get tokenizer from cache
        tokenizer = _get_tokenizer(model_name)
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            # Create messages format for chat template
            messages = [{"role": "user", "content": content}]
            
            # Apply the chat template
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted
        else:
            # Fallback to manual formatting if no chat template available
            return _manual_chat_format(content, model)
            
    except Exception as e:
        print(f"Warning: Could not use HF chat template for {model}: {e}")
        print("Falling back to manual formatting...")
        return _manual_chat_format(content, model)


def _manual_chat_format(content, model):
    """Fallback manual chat formatting when HF template is not available"""
    if model.lower() in ['llama', 'llama2', 'mistral']:
        return f"[INST] {content} [/INST]"
    elif model.lower() == 'vicuna':
        return f"USER: {content} ASSISTANT:"
    elif model.lower() in ['qwen', 'qwen2']:
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    elif model.lower() == 'llama3':
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model.lower() in ['olmo', 'olmo-instruct']:
        return f"<|user|>\n{content}\n<|assistant|>\n"
    else:
        return content


def read_attn(d):#for one entry
  attn=[np.array(arr) for arr in d['attentions']]
  #num_of_token, layer, head, seq,seq
  print(attn[0].shape)
  tokens_in=d['tokens_in']
  tokens_out=d['tokens_out']
  probs=d['probs']
  return attn,tokens_in,tokens_out,probs

def ret_top_attn(token_in,token_out,attn,pos,l,num_head=32):
  #l:decode token position
  seq=token_in+token_out
  ret=[]
  if pos==0:
    attn[0]=[[attn[0][l][h][-1].squeeze() for h in range(num_head)] for l in range(len(attn[0]))]

  mean_sort_idx=np.argsort(np.mean(attn[pos][l],axis=0))[-20:]
  v=np.sort(np.mean(attn[pos][l],axis=0))[-10:]

  for idx in mean_sort_idx:
    ret.append((idx,seq[idx]))
  return ret



def ret_topk_tok(probs,pos,k=10):
  sort_idx=np.argsort(probs[pos])[-k:]
  print(list(sort_idx))
  v=np.sort(probs[pos])[-k:]
  return sort_idx,v

if __name__ == '__main__':
    d=read_row('output/checkpoint_analysis_amc_deepseek-llama8b_right_no_perturb-3.jsonl')
    #data_size,layer,hidden_size
    print(len(d))
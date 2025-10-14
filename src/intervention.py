

import json
import math
import torch
import os
import copy
import argparse
from typing import List, Tuple, Callable, Optional, Union
from torch import Tensor
from tqdm import tqdm
from utils import read_row, formatInp, load_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import contextlib
import functools
import numpy as np
import random
import logging

# Constants
DECODING_STEP = 3
MODEL = 'llama'
COUNT_ADD=0


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
    return 4096  # Default fallback



@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    use_kwargs: bool = True,
    **kwargs
):
    """Context manager for adding and removing hooks from modules."""
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook, with_kwargs=use_kwargs))

        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def scale_attention_by_keys_pre_hook(positions, scale=1.0, num_heads=32,heads=None):
    """
    Bias attention *coefficients* toward specific KEY token positions.
    Multiplies p[:, h, q, k in positions] by `scale` (then renormalizes via softmax).

    Args:
      positions: list[int] key indices to boost/suppress globally (for all queries).
      scale: float > 0.  >1 boosts attention to those keys; <1 suppresses it.
      heads: None (all heads) or list[int] of head indices to affect.
    """
    if scale ==0:
        ln_alpha = float("-inf")
    else:
        ln_alpha = math.log(scale)

    def pre_hook(module, args, kwargs):
        # hidden_states: (B, T_q, D)
        global COUNT_ADD
        if not (DECODING_STEP == -1 or COUNT_ADD < DECODING_STEP):
            return args, kwargs
        
        if len(args) > 0:
            hidden_states = args[0]
        else:
            hidden_states = kwargs["hidden_states"]
        #print('hidden_states shape', hidden_states.shape)
        B, q_len, _ = hidden_states.shape

        # Determine heads and k_len (often from attention_mask if present)
        H = num_heads

        attn_mask = kwargs.get("attention_mask", None)
        k_len = attn_mask.size(-1) if attn_mask is not None else q_len  # fallback if no KV cache

        device, dtype = hidden_states.device, hidden_states.dtype
        bias = torch.zeros((B, H, q_len, k_len), device=device, dtype=dtype)

        # which heads to touch
        head_idx = range(H) if heads is None else heads

        # add log(scale) to the selected KEY columns
        for j in positions:
            if 0 <= j < k_len:
                bias[:, head_idx, :, j] += ln_alpha

        # merge with existing attention_mask
        #print('attention_mask before', kwargs["attention_mask"][:, head_idx, :, positions])
        if attn_mask is None:
            kwargs["attention_mask"] = bias
        else:
            kwargs["attention_mask"] = attn_mask + bias
        
        COUNT_ADD+=1
        return args, kwargs

    return pre_hook


def get_activation_addition_input_pre_hook(
    vector: Tensor, 
    coeff: Tensor,
    cache: Optional[List] = None,
    record: int = 0,
    intervene_all: bool = True,
    positions: List[int] = None
):
    """Hook function to add activation vectors to model inputs."""
    if cache is None:
        cache = []
    if positions is None:
        positions = [-1]

    def hook_fn(module, input):
        nonlocal vector, cache, coeff
        global COUNT_ADD
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
        device = activation.device  # get device of activation tensor
        vector = vector.to(device)  # move vector to same device
        coeff = coeff.to(device)  # move coeff to same device
        #print('vector shape', vector.shape)
        #print('activation shape', activation.shape)
        #print('DECODING_STEP', DECODING_STEP,'COUNT_ADD', COUNT_ADD)
        if DECODING_STEP == -1 or COUNT_ADD < DECODING_STEP:  # when equal to -1, till end of generation
            #assert coeff != 0
            if intervene_all:
                activation[:, positions[0]:, :] += coeff * vector
            else:
                for pos in positions:
                    #device = activation.device  # get device of activation tensor
                    #vector = vector.to(device)  # move vector to same device
                    activation[:, pos, :] += coeff * vector
            COUNT_ADD+=1
        if record:
            cache.append(activation)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_activation_addition_forward_hook(
    vector: Tensor,
    coeff: Tensor,
    cache: Optional[List] = None,
    record: int = 0,
    intervene_all: bool = True,
    positions: List[int] = None,
):
    """Forward hook variant of activation addition; modifies and returns the output."""
    if cache is None:
        cache = []
    if positions is None:
        positions = [-1]

    def hook_fn(module: torch.nn.Module, input, output):
        nonlocal vector, cache, coeff
        global COUNT_ADD
        activation = output[0] if isinstance(output, tuple) else output
        device = activation.device  # get device of activation tensor
        vector = vector.to(device)  # move vector to same device
        coeff = coeff.to(device)  # move coeff to same device

        if DECODING_STEP == -1 or COUNT_ADD < DECODING_STEP:
            assert coeff != 0
            if intervene_all:
                activation[:, positions[0]:, :] += coeff * vector
            else:
                # activation: [batch, seq, hidden]
                if activation.dim() == 2:
                    activation = activation.unsqueeze(0)
                activation_clone = activation.clone()
                for pos in positions:
                    activation_clone[:, pos, :] += coeff * vector
                activation = activation_clone
            COUNT_ADD += 1

        if record:
            cache.append(activation)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        return activation

    return hook_fn


def complete_with_intervention(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    instructions: List[dict], 
    tokenize_instructions_fn: Callable,
    intervene_layers: List[int], 
    batch_size: int = 32, 
    intervention_vector_ori: Optional[Tensor] = None,
    args: Optional[dict] = None
) -> List[dict]:
    """
    Complete text generation with intervention vectors.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer for the model
        instructions: List of instruction dictionaries
        tokenize_instructions_fn: Function to tokenize instructions
        intervene_layers: List of layer indices to intervene on
        batch_size: Batch size for processing
        intervention_vector_ori: Original intervention vectors
        args: Configuration arguments dictionary
        
    Returns:
        List of completion dictionaries containing generated text and metadata
    """
    coeff = torch.tensor(float(args['add_coef_intervene']))
    if args['intervene_all']:
        print('generate till the end of the model context window')
        generation_config = GenerationConfig(
            max_length=get_model_max_length(model),
            do_sample=False
        )
    else:
        generation_config = GenerationConfig(
            max_new_tokens=args['max_token_generate'], 
            do_sample=False
        )
    generation_config.pad_token_id = tokenizer.pad_token_id
    
    n_layers = model.config.num_hidden_layers
    logging.info(f"Number of layers: {n_layers}")

    ret = []
    cache = [[] for _ in range(n_layers)]
    global COUNT_ADD
    for i in tqdm(range(0, len(instructions), batch_size)):

        if intervention_vector_ori.shape[0] > 1:
            intervention_vector = intervention_vector_ori[i].squeeze()
        else:
            intervention_vector = intervention_vector_ori.squeeze()

        print('intervention vector shape', intervention_vector_ori.shape)
        print('intervention vector shape reshape', intervention_vector.shape)
        
        input_prompt,tokenized_instructions = tokenize_instructions_fn(instructions[i:i + batch_size])
        seq_len = tokenized_instructions.input_ids.shape[-1]
        # By default, intervene on all input tokens
        intervene_position = list(range(seq_len))
        # Optional: limit intervention to only the perturbed_step span
        if args.get('intervene_on_perturbed_step', 0) or args.get('intervene_on_jb_prompt', 0):
            try:
                if batch_size != 1:
                    print("Warning: intervene_position computed for first item only; batch_size>1 not fully supported")
                d0 = instructions[i]
                step_str = d0[args['step_key']]
                step_ids = tokenizer(step_str, add_special_tokens=False).input_ids
                full_ids = tokenized_instructions.input_ids[0].tolist()
                attn = tokenized_instructions.attention_mask[0].tolist()
                seq_len_eff = sum(attn)
                search_space = full_ids[:seq_len_eff]
                start_idx = None
                n = len(step_ids)
                for s in range(seq_len_eff - n, -1, -1):
                    if search_space[s:s + n] == step_ids:
                        start_idx = s
                        break
                if start_idx is None:
                    start_idx = max(0, seq_len_eff - n)
                intervene_position = list(range(start_idx, start_idx + n))
                print('intervene_position (perturbed_step):', intervene_position)
            except Exception as e:
                print(f"Failed to compute perturbed_step intervene positions: {e}")
                intervene_position = list(range(seq_len))

        # If context-only requested and not targeting perturbed_step, trim to context span
        if args['intervene_context_only'] and not args['intervene_on_perturbed_step']:
            if MODEL == 'qwen':
                special_token = "<|im_end|>\n<|im_start|>assistant"
            elif MODEL == 'llama3':
                special_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            elif MODEL == 'llama2':
                special_token = "[/INST]"
            special_token_len = len(tokenizer.tokenize(special_token))
            print('special token length', special_token_len)
            intervene_position = list(range(seq_len - special_token_len))
            
        print(50 * "#")
        print('intervene with steering vectors')
        print('intervene position', intervene_position) 
        
        fwd_pre_hooks = []
        fwd_hooks = []
        
        # Add attention head scaling hooks if enabled
        if args.get('scale_attention_heads', 0):
            print('Adding attention head scaling hooks')
            scale_factor = args.get('attention_scale_factor', 0.0)
            print(f'Attention scale factor: {scale_factor}')
            
            # Get model configuration for attention heads
            num_heads = getattr(model.config, 'num_attention_heads', 32)
            head_dim = getattr(model.config, 'hidden_size', 4096) // num_heads
            
            # Apply attention scaling to all specified layers
            for intervene_layer in intervene_layers:
                if intervene_layer < model.config.num_hidden_layers:
                    # Apply to self-attention layer - hook to the main attention module
                    attention_module = model.model.layers[intervene_layer].self_attn
                    fwd_pre_hooks.append(
                        (
                            attention_module,
                            scale_attention_by_keys_pre_hook(
                                positions=intervene_position,
                                scale=scale_factor,
                                num_heads=num_heads,
                            )
                        )
                    )
        
        for intervene_layer in intervene_layers:
            if args['scale_attention_heads']:
                break #only scale attention heads hardcoded for now
            if intervene_layer == n_layers:
                # use forward hook on last layer
                fwd_hooks.append(
                    (
                        model.model.layers[intervene_layer-1],
                        get_activation_addition_forward_hook(
                            vector=intervention_vector[intervene_layer, :],
                            coeff=coeff,
                            intervene_all=args['intervene_all'],
                            positions=intervene_position,
                        ),
                    )
                )
            else:
                # use input pre-hook on other layers
                fwd_pre_hooks.append(
                    (
                        model.model.layers[intervene_layer],
                        get_activation_addition_input_pre_hook(
                            vector=intervention_vector[intervene_layer, :],
                            coeff=coeff,
                            intervene_all=args['intervene_all'],
                            positions=intervene_position,
                        ),
                    )
                )

        completions = []

        print('index', i)
        print('length of instructions', len(instructions))
        assert i < len(instructions)

        if not args['intervene_all']:
            for id in intervene_position:
                print('intervening token', tokenizer.convert_ids_to_tokens(tokenized_instructions.input_ids[0])[id])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks,use_kwargs=args['scale_attention_heads']):
            if 0:
                # Use direct model() call for attention head scaling with token generation loop
                print("Using direct model() call for attention head scaling")
                max_new_tokens = args['max_token_generate']
                current_input_ids = tokenized_instructions.input_ids.to(model.device)
                current_attention_mask = tokenized_instructions.attention_mask.to(model.device)
                
                generated_tokens = []
                generated_scores = []
                prob = []
                tokens = []
                
                with torch.no_grad():
                    for step in range(max_new_tokens):
                        outputs = model(
                            input_ids=current_input_ids,
                            attention_mask=current_attention_mask,
                            output_attentions=True,
                            return_dict=True
                        )
                        
                        # Get logits for the last token position
                        logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                        generated_scores.append(logits)
                        
                        # Get the most likely next token
                        next_token_id = torch.argmax(logits, dim=-1)  # Shape: [batch_size]
                        generated_tokens.append(next_token_id)
                        
                        # Record probabilities if requested
                        if args['record_probs'] and step < 10:
                            probability = torch.softmax(logits[0], dim=-1)
                            all_probs = probability.detach().cpu().numpy().tolist()
                            token_id = next_token_id[0].item()
                            token = tokenizer.decode(token_id)
                            prob.append(all_probs)
                            tokens.append(token)
                        
                        # Update input for next iteration
                        current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token_id.unsqueeze(-1))], dim=-1)
                        
                        # Check for EOS token
                        if tokenizer.eos_token_id is not None and next_token_id[0].item() == tokenizer.eos_token_id:
                            print(f"EOS token generated at step {step}")
                            break
                
                # Combine and decode generated tokens
                if generated_tokens:
                    generation_token_ids = torch.cat(generated_tokens, dim=-1)  # Shape: [batch_size, num_generated]
                    print('generation_token_ids shape', generation_token_ids.shape)
                    # Decode tokens to match the format expected by the rest of the code
                    generation_toks = []
                    for batch_idx in range(generation_token_ids.shape[0]):
                        decoded_text = tokenizer.decode(generation_token_ids[batch_idx], skip_special_tokens=True)
                        generation_toks.append(decoded_text)
                else:
                    generation_toks = [""] * current_input_ids.shape[0]  # Empty strings for each batch item
                
                print('length of input token', tokenized_instructions.input_ids.shape[-1])
                if generation_toks:
                    print('generated text length:', len(generation_toks[0]) if generation_toks[0] else 0)
                
            else:
                # Use standard generation for activation vector interventions
                generation_toks = model.generate(
                    input_ids=tokenized_instructions.input_ids.to(model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(model.device),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=True,
                    use_cache= args['max_decode_step_while_intervene']==1
                )
                    
                generated_scores = generation_toks.scores
                prob = []
                tokens = []
                
                if args['record_probs']:
                    for ii, score in enumerate(generated_scores):
                        if ii < 10:
                            probability = torch.softmax(score[0], dim=-1)
                            all_probs = probability.detach().cpu().numpy().tolist()
                            token_id = torch.argmax(probability).item()
                            prob_id = probability[token_id].item()
                            token = tokenizer.decode(token_id)
                            prob.append(all_probs)
                            tokens.append(token)
                        else:
                            break
                            
                print('length of input token', tokenized_instructions.input_ids.shape[-1])
                generation_toks = generation_toks.sequences[:, tokenized_instructions.input_ids.shape[-1]:]  # remove input tokens
                print('length of generated token', generation_toks.shape[-1])

            generation_idx, generation =0,generation_toks[0]

            response_text = tokenizer.decode(generation, skip_special_tokens=True).strip()
            
            completions.append({
                'input_prompt': input_prompt,
                'prompt': instructions[i + generation_idx],
                'response': response_text,
                'tokens': tokens,
                'probs': prob,
            })
            COUNT_ADD=0
        ret.extend(completions)
        # optional: emit minimal cache info if used elsewhere
            
    return ret


def main():
    """
    Run the full text generation pipeline with intervention vectors.
    
    This function sets up the model, loads intervention vectors, and performs
    controlled text generation with various intervention strategies.
    """
    parser = argparse.ArgumentParser(description="Text generation with intervention vectors")
    
    # Model configuration
    parser.add_argument("--model", default='llama', type=str, help="Model type")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    # Data paths
    parser.add_argument("--test_data_pth", default='data/medcq.json', type=str, help='Test data path')
    parser.add_argument("--output_pth", default='data/medcq.json', type=str, help="Output file path")
    parser.add_argument("--intervention_vector", default=None, type=str, help='Intervention vector path')
    
    # Processing parameters
    parser.add_argument('--left', default=0, type=int, help='Left index')
    parser.add_argument('--right', default=10, type=int, help='Right index')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--mode', default='complete', type=str, help='Mode')
    
    # Intervention parameters
    parser.add_argument('--add_coef_intervene', default=1., type=float, help='Intervention coefficient')
    parser.add_argument('--layer_s', default=0, type=int, help='Start layer')
    parser.add_argument('--layer_e', default=32, type=int, help='End layer')
    parser.add_argument('--intervention_vector_layer', default=-1, type=int, help='Intervention vector at a layer')
    parser.add_argument('--reverse_intervention', default=0, type=int, help='Reverse intervention')
    parser.add_argument('--intervene_all', default=0, type=int, help='Intervene all tokens')
    parser.add_argument('--intervene_context_only', default=0, type=int, help='Intervene context only before the inversion prompt, used for reply inversion task only')
    
    # Generation parameters
    parser.add_argument('--max_token_generate', default=400, type=int, help='Max tokens to generate')
    parser.add_argument('--max_decode_step_while_intervene', default=1, type=int, help='Max decode steps while intervening')

    # Recording parameters
    parser.add_argument('--record_probs', default=0, type=int, help='Record probabilities')
 
    # Model-specific parameters
    parser.add_argument('--remove_inst_inp', default=0, type=int, help='Remove instruction input')
    parser.add_argument('--use_inversion', default=0, type=int, help='Use inversion reply task')
    parser.add_argument('--inversion_prompt_idx', default=0, type=int, help='Inversion prompt index')
    parser.add_argument('--mask_attn_token', default=0, type=int, help='Mask attention tokens')
    # Used parameters
    parser.add_argument('--use_jailbreak_test', default=0, type=int, help='Use the jailbreak version of inputs at test')
    parser.add_argument('--coeff_select', default=1, type=float, help='Coefficient select')
    parser.add_argument('--traverse_single_layer_intervention', default=0, type=int, help='Traverse single layer intervention')
    parser.add_argument('--load_ckpt', default=0, type=int, help='Load checkpoint')
    parser.add_argument('--peft_pth_ckpt', default='output/attn.json', type=str, help='PEFT checkpoint path')
    parser.add_argument('--positions', default='-1', type=str, help='Positions')
    parser.add_argument('--intervene_on_perturbed_step', default=0, type=int, help='Intervene on perturbed step')
    parser.add_argument('--step_key', default='perturb_step', type=str, help='Key to use for accessing step data (e.g., initial_step, perturb_step, consistent_step)')
    parser.add_argument('--include_final_answer_cue', default=1, type=int, help='Whether to append final answer cue to prompt')
    parser.add_argument('--context_key', default='context_text', type=str, help='Key to use for accessing context data')
    parser.add_argument('--intervene_on_jb_prompt', default=0, type=int, help='Intervene on jailbreak prompt')
    parser.add_argument('--omit_thinking', default=0, type=int, help='Omit thinking')
    parser.add_argument('--scale_attention_heads', default=0, type=int, help='Scale attention heads intervention')
    parser.add_argument('--attention_scale_factor', default=0.0, type=float, help='Scale factor for attention heads')
    args = parser.parse_args()
    params = vars(args)
    
    global MODEL, DECODING_STEP
    DECODING_STEP = params['max_decode_step_while_intervene'] #if -1, since the input token, till end of generation
    MODEL = params['model']
    
    # Parse list arguments
    params['positions'] = list(map(int, params['positions'].split()))
    

    # Model loading with error handling
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL, params['model_size'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if params['load_ckpt']:
        print('loading ckpt adapter')
        try:
            model.load_adapter(params['peft_pth_ckpt'])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

    if params['mode'] == 'complete':
        try:
            test_data = read_row(params['test_data_pth'])[params['left']:params['right']]
        except Exception as e:
            print(f"Error loading test data: {e}")
            return
            
        try:
            intervention_vector = torch.load(params['intervention_vector'], weights_only=False)
        except Exception as e:
            print(f"Error loading intervention vector: {e}")
            return

        candidate_coeff = [params['coeff_select']] #if params['coeff_select'] else [1] 
        
        for coeff in candidate_coeff:
            params['add_coef_intervene'] = coeff
            print('coefficient for intervention', coeff)
            
            layer_list = list(range(13, 32)) if params['traverse_single_layer_intervention'] else [0]
            
            for out_i in layer_list:
                for i in range(params['layer_s'], params['layer_e']):
                    print('##################')
                    print('intervention layer', i)                        
                    use_persuade = params['use_jailbreak_test']
                    if isinstance(intervention_vector, np.ndarray):
                        intervention_vector = torch.tensor(intervention_vector)
                    if len(intervention_vector.shape) == 2:
                        intervention_vector = intervention_vector.unsqueeze(0)
                        
                    print('intervention vector shape', intervention_vector.shape)

                    if params['intervention_vector_layer'] >= 0:
                        intervention_vector_local = intervention_vector[:, params['intervention_vector_layer'], :].to(model.device)
                    elif params['traverse_single_layer_intervention']:
                        intervention_vector_local = intervention_vector[:, out_i, :].to(model.device)
                    else:
                        if intervention_vector.shape[0] != 1:
                            intervention_vector_local = intervention_vector[params['left']:params['right']].to(model.device)
                        else:
                            intervention_vector_local = intervention_vector.to(model.device)

                    if params['reverse_intervention']:
                        intervention_vector_local = -intervention_vector_local

                    def tokenize_instructions_fn(instructions, include_final_answer_cue=params['include_final_answer_cue'], intervene_on_perturbed_step=params['intervene_on_perturbed_step']):
                        final_answer_cue = '\n The final answer is \\boxed{' if include_final_answer_cue else ''
                        step_key = params['step_key']
                        context_key = params['context_key'] #by default, context_text
                        if intervene_on_perturbed_step:
                            inps = [
                                formatInp(i, model=MODEL, use_template=True, tokenizer=tokenizer)+'. '+i[context_key] +'. '+i[step_key]+final_answer_cue
                                for i in instructions
                            ]
                        else:  
                            inps = [
                                formatInp(i, model=MODEL, use_template=True, tokenizer=tokenizer)
                                for i in instructions
                            ]
                        if params['omit_thinking']:
                            inps=[ inp + '\n\n</think>' for inp in inps]
                        return inps,tokenizer(inps, padding=True, return_tensors="pt")

                    
                    completions = complete_with_intervention(
                        model, tokenizer, test_data, tokenize_instructions_fn,
                        [i], batch_size=params['batch_size'], 
                        intervention_vector_ori=intervention_vector_local,
                        args=params,
                    )
                        
                    # Save results
                    output_file = params['output_pth'].replace('.json', f'-intervene{i}.json')
                    with open(output_file, 'w') as f:
                        for completion in completions:
                            json.dump(completion, f)
                            f.write("\n")
                            



if __name__ == "__main__":
    main()
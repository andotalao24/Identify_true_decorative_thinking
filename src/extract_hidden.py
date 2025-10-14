import json
import torch
import os
import copy
import argparse
from typing import List, Tuple, Callable, Optional
from torch import Tensor
from tqdm import tqdm
from utils import read_row, formatInp, load_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib
import functools
import random
import logging

logger = logging.getLogger(__name__)


MODEL = ''
NUM_TOKEN_HIDDEN = 0  # by default, we extract NUM_TOKEN_HIDDEN tokens + all special post-instruction tokens

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
) -> None:
    """
    Context manager for temporarily adding forward hooks to a model.

    Args:
        module_forward_pre_hooks: A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
        module_forward_hooks: A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
        **kwargs: Additional keyword arguments to pass to the hooks
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))

        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_mean_activations_pre_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN
) -> Callable:
    """
    Creates a hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...]) -> None:
        activation = input[0].half()
        
        # Handle both 2D and 3D tensors
        if len(activation.shape) == 2:
            # For 2D tensors, assume (seq_len, hidden_dim) and add batch dimension
            activation = activation.unsqueeze(0)
            
        seq_len = activation.shape[1]
        
        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                
                assert isinstance(positions[0], int)
                context = activation[:, -len(positions)-step:-len(positions), :]
                pos_activations = activation[:, positions, :]
                merged_activation = torch.cat([context, pos_activations], dim=1)
                cache_full[layer].append(merged_activation.clone().detach().cpu())
            else:
                print('seq_len<positions', seq_len, len(positions))
                exit()
    return hook_fn

def get_mean_activations_fwd_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN
) -> Callable:
    """
    Creates a forward hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...], output: Tuple[Tensor, ...]) -> None:
        activation = output[0].half()
        print('activation shape', activation.shape)
        
        # Handle both 2D and 3D tensors
        if len(activation.shape) == 2:
            # For 2D tensors, assume (seq_len, hidden_dim) and add batch dimension
            activation = activation.unsqueeze(0)
            
        seq_len = activation.shape[1]
        
        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                context = activation[:, -len(positions)-step:-len(positions), :]
                pos_activations = activation[:, positions, :]
                merged_activation = torch.cat([context, pos_activations], dim=1)
                cache_full[layer].append(merged_activation.clone().detach().cpu())
            else:
                print('seq_len<positions', seq_len, len(positions))
                exit()
    return hook_fn


def get_mean_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    tokenize_instructions_fn: Callable,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
    ret_whole_seq: bool = False,
    use_perturbed_step: bool = False,
    args: dict = None
) -> Tuple[Tensor, Tensor]:
    """
    Extracts mean activations from model for given instructions.

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer instance
        instructions: List of input instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model blocks to hook
        batch_size: Batch size for processing
        positions: Positions to extract activations from
        ret_whole_seq: Whether to return whole sequence

    Returns:
        Tuple of (mean activations, full activations)
    """
    torch.cuda.empty_cache()

    n_layers = model.config.num_hidden_layers
    full_activations = [[] for _ in range(n_layers + 1)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i:i+batch_size]
        inputs = tokenize_instructions_fn(instructions=batch_instructions)

        # Determine positions for this batch
        positions_local = positions
        if use_perturbed_step:
            logger.info(f"Extracting positions from perturbed step")
            if batch_size != 1:
                print("Warning: positions computed for first item only; batch_size>1 not fully supported")
            d0 = batch_instructions[0]
            step_key = (args or {}).get('step_tag', 'perturb_step')
            step_str = d0[step_key]
            # Tokenize step without forcing a leading space
            step_ids = tokenizer(step_str, add_special_tokens=False).input_ids
            full_ids = inputs.input_ids[0].tolist()
            attn = inputs.attention_mask[0].tolist()
            seq_len_eff = sum(attn)
            search_space = full_ids[:seq_len_eff]
            start_idx = None
            n = len(step_ids)
            # Find last occurrence of step_ids (as step is appended at the end)
            for s in range(seq_len_eff - n, -1, -1):
                if search_space[s:s + n] == step_ids:
                    start_idx = s
                    break
            # Fallback: also try with a leading space tokenization if not found
            if start_idx is None:
                step_ids_space = tokenizer(' ' + step_str, add_special_tokens=False).input_ids
                n2 = len(step_ids_space)
                for s in range(seq_len_eff - n2, -1, -1):
                    if search_space[s:s + n2] == step_ids_space:
                        start_idx = s
                        n = n2
                        step_ids = step_ids_space
                        break
            assert start_idx is not None
            positions_local = list(range(start_idx, start_idx + n))
            print('extract positions (perturbed_step):', positions_local)

        logger.info(f"Extracting positions: {positions_local}")
        
        # Create hooks with positions for this batch
        fwd_pre_hooks = [
            (block_modules[layer], get_mean_activations_pre_hook(
                layer=layer,
                cache_full=full_activations,
                positions=positions_local,
                whole_seq=ret_whole_seq
            )) for layer in range(n_layers)
        ]
        
        fwd_hooks = [
            (block_modules[n_layers-1], get_mean_activations_fwd_hook(
                layer=-1,
                cache_full=full_activations,
                positions=positions_local,
                whole_seq=ret_whole_seq
            ))
        ]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    try:
        # Fast path: all position lengths equal across examples
        flat_list = [torch.stack(inner_list) for inner_list in full_activations]
        result = torch.stack(flat_list).squeeze()
        if len(result.shape) < 3:
            result = result.unsqueeze(1)
        mean_activations = result.mean(dim=1)
    except Exception as e:
        # Fallback for variable-length positions: average over positions then over examples
        per_layer_bh = []  # list of [num_examples, hidden]
        for inner_list in full_activations:
            bh_list = []
            for t in inner_list:
                # Ensure 3D: [batch, pos, hidden]
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                # Average over positions
                bh_list.append(t.mean(dim=1))  # [batch, hidden]
            if len(bh_list) == 0:
                per_layer_bh.append(torch.empty(0, 0))
            else:
                per_layer_bh.append(torch.cat(bh_list, dim=0))  # [num_examples, hidden]

        # Stack layers along dim 0; assume same num_examples across layers
        
        result = torch.stack(per_layer_bh, dim=0)  # [layers, num_examples, hidden]

        # Mean over examples, ignoring NaNs if any padding occurred
        if torch.isnan(result).any():
            mean_activations = torch.nanmean(result, dim=1)
        else:
            mean_activations = result.mean(dim=1)

    print('mean shape', mean_activations.shape)
    return mean_activations, result



def get_mean_diff(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    tokenize_instructions_fn: Callable,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
    extract_only: bool = False,
    use_persuade_harmful: bool = False,
    use_persuade_harmless: bool = False,
    use_sys_harmful: bool = False,
    ret_whole_seq: bool = False,
    use_perturbed_step: bool = False,
    args: dict | None = None
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Computes mean activation differences between harmful and harmless instructions.

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model blocks to hook
        batch_size: Batch size for processing
        positions: Positions to extract activations from
        extract_only: Whether to only extract harmful activations
        use_persuade_harmful: Whether to use persuasion for harmful
        use_persuade_harmless: Whether to use persuasion for harmless
        use_sys_harmful: Whether to use system prompt for harmful
        ret_whole_seq: Whether to return whole sequence

    Returns:
        Tuple of (harmful mean activations, harmless mean activations,
                harmful full activations, harmless full activations)
    """
    mean_activations_harmful, full_activations_harmful = get_mean_activations(
        model, tokenizer, harmful_instructions,
        functools.partial(tokenize_instructions_fn, use_persuade=use_persuade_harmful, use_sys=use_sys_harmful),
        block_modules, batch_size=batch_size, positions=positions, ret_whole_seq=ret_whole_seq, use_perturbed_step=use_perturbed_step, args=args
    )
    
    torch.save(mean_activations_harmful, 'output/tmp_mean_activations_harmful.pt')
    torch.save(full_activations_harmful, 'output/tmp_full_activations_harmful.pt')
    del mean_activations_harmful, full_activations_harmful
    torch.cuda.empty_cache()

    if not extract_only:
        mean_activations_harmless, full_activations_harmless = get_mean_activations(
            model, tokenizer, harmless_instructions,
            functools.partial(tokenize_instructions_fn, use_persuade=use_persuade_harmless),
            block_modules, batch_size=batch_size, positions=positions, ret_whole_seq=ret_whole_seq, use_perturbed_step=use_perturbed_step, args=args
        )
        mean_activations_harmful = torch.load('output/tmp_mean_activations_harmful.pt', weights_only=False)
        full_activations_harmful = torch.load('output/tmp_full_activations_harmful.pt', weights_only=False)

        print('mean_activations_harmful shape', mean_activations_harmful.shape)
        print('mean_activations_harmless shape', mean_activations_harmless.shape)
    else:
        mean_activations_harmless = None
        full_activations_harmless = None

    return mean_activations_harmful, mean_activations_harmless, full_activations_harmful, full_activations_harmless

def generate_directions(
    model_base: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    args: dict
) -> Optional[Tensor]:
    """
    Generates direction vectors from model activations.

    Args:
        model_base: Base model
        tokenizer: Tokenizer instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        args: Arguments dictionary

    Returns:
        Mean difference tensor or None if computation fails
    """
    def tokenize_instructions_fn(instructions: List[str], use_persuade: bool = False, use_sys: bool = False) -> dict:
        #include the ori_output in the prompt
        inps = [formatInp(i, model=MODEL, use_template=True, tokenizer=tokenizer)+'. '+i['context_text'] +'. '+i[args['step_tag']]
                             for i in instructions
                        ]
        return tokenizer(inps, padding=True, return_tensors="pt")

    model_block_modules = model_base.model.layers
    mean_activations_harmful, mean_activations_harmless, all_harmful, all_harmless = get_mean_diff(
        model_base, tokenizer, harmful_instructions, harmless_instructions,
        tokenize_instructions_fn, model_block_modules, args['batch_size'],
        args['positions'], args['extract_only'], args['use_persuade_harmful'],
        args['use_persuade_harmless'], args['use_sys_harmful'], args['ret_whole_seq'],
        args['extract_from_perturbed_step'], args
    )

    torch.save(all_harmful, args['output_pth_harmful'])
    torch.save(all_harmless, args['output_pth_harmless'])

    try:
        print('mean_activations_harmful shape', mean_activations_harmful.shape)
        print('mean_activations_harmless shape', mean_activations_harmless.shape)
        mean_diffs = mean_activations_harmful - mean_activations_harmless
        assert not mean_diffs.isnan().any()
        #if args['mode_dir'] == 'hf':
            #mean_diffs = mean_diffs[:,NUM_TOKEN_HIDDEN-1] 
        #elif args['mode_dir'] == 'refuse':
            #mean_diffs = mean_diffs[:,-1]
        torch.save(mean_diffs.to('cpu'), args['output_pth'])
    except Exception as e:
        print(e)
        mean_diffs = None

    return mean_diffs

def main() -> None:
    """Run the full pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='llama', type=str, help="Model type")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size")
    parser.add_argument("--harmful_pth", default='data/medcq.json', type=str, help="Path to harmful examples")
    parser.add_argument("--harmless_pth", default='data/medcq.json', type=str, help="Path to harmless examples")
    parser.add_argument("--output_pth_harmful", default='output/mean_diff.pt', type=str, help="Output path for harmful activations")
    parser.add_argument("--output_pth_harmless", default='output/mean_diff_harmless.pt', type=str, help="Output path for harmless activations")
    parser.add_argument('--use_persuade_harmful', default=0, type=int, help='Use persuasion for harmful examples')
    parser.add_argument('--use_persuade_harmless', default=0, type=int, help='Use persuasion for harmless examples')
    parser.add_argument('--use_sys_harmful', default=0, type=int, help='Use system prompt for harmful examples')
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=10, type=int, help='Right index for data slicing')
    parser.add_argument('--random_sample_harmful', default=0, type=int, help='Randomly sample harmful examples')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument("--output_pth", default='output/dir.pt', type=str, help="Output path of generated directions")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument('--mode', default='diff-mean', type=str, help='Mode')
    parser.add_argument('--positions', default='-1', type=str, help='Positions to extract')
    parser.add_argument('--extract_only', default=0, type=int, help='Only extract harmful activations')
    parser.add_argument('--ret_whole_seq', default=0, type=int, help='Return whole sequence')
    parser.add_argument('--extract_hidden_inst_token', default=0, type=int, help="Extract hidden state of instruction tokens")
    parser.add_argument('--extract_harmful_token_only', default=0, type=int, help="Extract harmful token only")
    parser.add_argument('--mode_dir', default='hf', type=str, help="Mode for direction extraction: 'hf' or 'refuse'")
    parser.add_argument('--use_template', default=1, type=int, help="Whether to use chat template formatting")
    parser.add_argument('--extract_from_perturbed_step', default=0, type=int, help='Extract hidden states only from perturbed_step tokens')
    parser.add_argument('--step_tag', default='perturb_step', type=str, help='Tag for perturbed step')

    args = parser.parse_args()
    params = vars(args)
    
    global MODEL
    global NUM_TOKEN_HIDDEN
    
    params['positions'] = list(map(int, params['positions'].split()))
    MODEL = params['model']
    model, tokenizer = load_model_and_tokenizer(params['model'], params['model_size'])
    
    params['positions'] = [-1]

    harmful_train = read_row(params['harmful_pth'])

    if params['random_sample_harmful']:
        random.seed(params['left'] % len(harmful_train))
        harmful_train = random.sample(harmful_train, 1)
    else:
        if params['left'] < len(harmful_train):
            harmful_train = harmful_train[params['left']:params['right']]
        else:
            harmful_train = harmful_train[params['left'] % len(harmful_train):params['left'] % len(harmful_train)+1]
            
    harmless_train = read_row(params['harmless_pth'])[params['left']:params['right']]

    with open(params['output_pth'].replace('.pt', '_prompts_used.json'), 'w') as f:
        json.dump({'harmful': harmful_train, 'harmless': harmless_train}, f, indent=4)

    candidate_directions = generate_directions(model, tokenizer, harmful_train, harmless_train, params)
    print(candidate_directions.shape)

if __name__ == "__main__":
    main()
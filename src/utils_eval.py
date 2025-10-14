"""Evaluation utilities for checkpoint analysis.

This module provides utilities for evaluating language model outputs,
including answer comparison, text perturbation, and number replacement
strategies for causal analysis.
"""

import json
import logging
import random
import re
from typing import Optional, Tuple

import numpy as np

# Set up logging for this module
logger = logging.getLogger(__name__)



def compare_answers(extracted_ans: str, correct_ans: str) -> bool:
    """Compare extracted answer with correct answer.
    
    This function performs multiple types of comparison:
    - Direct string comparison
    - Numerical comparison with tolerance
    - Fraction comparison (e.g., "1/2" vs "0.5")
    
    Args:
        extracted_ans: The answer extracted from model output
        correct_ans: The correct answer to compare against
        
    Returns:
        True if answers match, False otherwise
    """
    # Clean extracted answer
    extracted_ans = str(extracted_ans).split("$")[0].split("\n")[0]
    for sign in [',', ':', '%']:
        extracted_ans = extracted_ans.replace(sign, '')
    extracted_ans = str(extracted_ans).strip()
    correct_ans = str(correct_ans).strip()
    
    # Direct string comparison
    if extracted_ans == correct_ans:
        return True
    
    # Try numerical comparison
    try:
        extracted_float = float(extracted_ans)
        correct_float = float(correct_ans)
        if abs(extracted_float - correct_float) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    
    # Try fraction comparison (e.g., "1/2" vs "0.5")
    try:
        if '/' in extracted_ans:
            num, denom = extracted_ans.split('/')
            extracted_float = float(num) / float(denom)
            correct_float = float(correct_ans)
            if abs(extracted_float - correct_float) < 1e-6:
                return True
        elif '/' in correct_ans:
            num, denom = correct_ans.split('/')
            correct_float = float(num) / float(denom)
            extracted_float = float(extracted_ans)
            if abs(extracted_float - correct_float) < 1e-6:
                return True
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return False



def rephrase_with_gpt4(text: str, task: str = "rephrase") -> str:
    """Use GPT-4 to transform the text.
    
    Args:
        text: Input text to transform
        task: Transformation type:
            - "rephrase": preserve mathematical content and numbers, change wording only
            - "distort": deliberately distort logic/expressions to be incorrect or unrelated
        
    Returns:
        Transformed text, or original text if API call fails
        
    Raises:
        ImportError: If required dependencies are not available
    """
    try:
        # Import API key and client locally to avoid hard dependency
        try:
            from key import KEY
            from openai import OpenAI
        except ImportError as e:
            logger.warning(f"Failed to import required dependencies: {e}")
            return text

        client = OpenAI(api_key=KEY)

        if task == "distort":
            system_instruction = (
                "You will rewrite the user's mathematical reasoning so that it becomes logically incorrect, "
                "misleading, or unrelated. You may alter equations, steps, and relationships, and introduce contradictions. "
                "Produce fluent text, but ensure the reasoning is wrong or off-topic. Do not include meta commentary."
            )
            user_instruction = (
                "Rewrite the following text by distorting the logic or mathematical expressions so the result is incorrect "
                "or unrelated. You can change numbers, steps, and relationships. Return only the distorted text:\n\n"
            )
            temperature = 0.9
        else:
            system_instruction = (
                "You will rephrase the user's mathematical reasoning while preserving all numbers, expressions, and the logical flow."
            )
            user_instruction = (
                "Please rephrase the following mathematical reasoning text while preserving all numbers, mathematical expressions, "
                "and the logical flow. Keep the mathematical content exactly the same but vary the wording and sentence structure:\n\n"
            )
            temperature = 0.7

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction + text},
            ],
            max_tokens=max(64, len(text.split()) * 2),
            temperature=temperature,
        )

        transformed = response.choices[0].message.content.strip()
        logger.info(f"Successfully transformed text using GPT-4 (task={task})")
        logger.debug(f"Original: {text}")
        logger.debug(f"Transformed: {transformed}")
        return transformed

    except Exception as e:
        logger.warning(f"Failed to transform text with GPT-4 (task={task}): {e}")
        logger.warning("Returning original text")
        return text




def randomly_replace_numbers(
    text: str, 
    random_seed: Optional[int] = None, 
    mode: str = 'random', 
    replace_all: bool = True, 
    rephrase_with_gpt: bool = False, 
    model=None, 
    tokenizer=None
) -> Tuple[str, bool]:
    """Randomly replace numbers in text for perturbation analysis.
    
    Args:
        text: Input text to perturb
        random_seed: Seed for reproducible randomization
        mode: Replacement mode ('random', 'random_add_small', etc.)
        replace_all: Whether to replace all numbers or just one
        rephrase_with_gpt: Whether to use GPT for rephrasing
        model: Model for embedding-based replacement
        tokenizer: Tokenizer for embedding-based replacement
        
    Returns:
        Tuple of (modified_text, was_modified)
    """
    local_random = random.Random(random_seed)
    logger.debug(f"Random seed set to {random_seed} for number replacement")

    if mode == 'gpt4_distort':
        distorted_text = rephrase_with_gpt4(text, task="distort")
        return distorted_text, distorted_text != text

    is_digit_pattern = r'is\s+(\d+(?:\.\d+)?)'
    equal_digit_pattern = r'=\s*(\d+(?:\.\d+)?)'
    
    is_digit_matches = list(re.finditer(is_digit_pattern, text))
    equal_digit_matches = list(re.finditer(equal_digit_pattern, text))
    if replace_all:
        # Replace all numbers robustly using a single regex substitution with a callback
        number_pattern = re.compile(r'[-+]?\d+(?:\.\d+)?')

        def _repl(m: re.Match) -> str:
            number_str = m.group(0)
            try:
                original_number = float(number_str)
                new_number = replace_new_number(original_number, mode=mode, random_generator=local_random, model=model, tokenizer=tokenizer)
                if isinstance(new_number, str):
                    return new_number
                if '.' in number_str:
                    return f"{new_number:.1f}" if (new_number % 1) != 0 else f"{int(new_number)}.0"
                return str(int(new_number))
            except Exception:
                # On any failure, keep original
                return number_str

        replaced_text = number_pattern.sub(_repl, text)
        if replaced_text == text:
            logger.info("No numbers found in text")
            return text, False
        return replaced_text, True
    # Combine all matches and find the one at the latest position
    all_matches = []
    for match in is_digit_matches:
        all_matches.append(('is', match))
    for match in equal_digit_matches:
        all_matches.append(('=', match))
    
    if all_matches:
        # Original logic: replace only the latest match
        all_matches.sort(key=lambda x: x[1].start())
        pattern_type, match_to_replace = all_matches[-1]
        
        number_to_replace = match_to_replace.group(1)  # The number part (group 1)
        
        original_number = float(number_to_replace)
        new_number = replace_new_number(original_number, mode=mode, random_generator=local_random, model=model, tokenizer=tokenizer)
        
        # Preserve format: integer stays integer, decimal stays decimal
        if '.' in number_to_replace:
            new_number_str = f"{new_number:.1f}" if new_number % 1 != 0 else f"{int(new_number)}.0"
        else:
            new_number_str = str(int(new_number))
        
        # Replace only the number part, keeping the pattern and any whitespace intact
        start_pos = match_to_replace.start(1)  # Start of the number group
        end_pos = match_to_replace.end(1)      # End of the number group
        
        modified_text = text[:start_pos] + new_number_str + text[end_pos:]
        
        #logger.debug(f"Added offset {offset} to number '{number_to_replace}' -> '{new_number_str}' after '{pattern_type}' at position {start_pos}-{end_pos}")
        return modified_text, True
    
    # Step 2: If no numbers found after "is" or "=", randomly select number(s) in the text
   
    else:
        # Original logic: replace only one random number by adding offset
        number_matches = list(re.finditer(r'\d+(?:\.\d+)?', text))
        
        if not number_matches:
            logger.debug("No numbers found in text")
            return text, False  # No numbers found, return original text
        
        # Randomly select one number to modify from anywhere in the text
        match_to_replace = local_random.choice(number_matches)
        number_to_replace = match_to_replace.group()
        
        original_number = float(number_to_replace)
        new_number = replace_new_number(
            original_number, mode=mode, random_generator=local_random, 
            model=model, tokenizer=tokenizer
        )
        
        # Preserve format: integer stays integer, decimal stays decimal
        if '.' in number_to_replace:
            new_number_str = f"{new_number:.1f}" if new_number % 1 != 0 else f"{int(new_number)}.0"
        else:
            new_number_str = str(int(new_number))
        
        # Replace the selected number
        start_pos = match_to_replace.start()
        end_pos = match_to_replace.end()
        modified_text = text[:start_pos] + new_number_str + text[end_pos:]
        
        return modified_text, True

        
def replace_new_number(
    old_number: float, 
    mode: str = 'random', 
    random_generator=None, 
    model=None, 
    tokenizer=None
) -> float:
    """Replace a number based on the specified mode.
    
    Args:
        old_number: The original number to replace
        mode: The replacement mode ('random', 'random_add_small', 'random_add_large', 
             'close_embed', 'far_embed')
        random_generator: Optional random generator to use (for seeded randomness)
        model: Model for embedding modes (required for close_embed/far_embed)
        tokenizer: Tokenizer for embedding modes (required for close_embed/far_embed)
    
    Returns:
        New number based on the specified mode
    """
    # Use provided random generator or fall back to global random
    rng = random_generator if random_generator is not None else random
    
    if mode == 'random':
        # Use list comprehension to avoid invalid list subtraction
        choices = [x for x in range(-1000, 1000) if x != old_number]
        new_number = rng.choice(choices)

    elif mode == 'random_add_small':
        # Use list comprehension to avoid invalid list subtraction
        offset_choices = [x for x in range(-3, 3) if x != 0]
        new_number = old_number + rng.choice(offset_choices)
        logger.debug(f"Added offset {new_number-old_number} to number '{old_number}' -> '{new_number}'")
    elif mode == 'random_add_large':
        new_number = old_number + rng.choice([-9999, -999, -99, 99, 999, 9999])
        logger.debug(f"Added offset {new_number-old_number} to number '{old_number}' -> '{new_number}'")
    
    elif mode == 'close_embed':
        assert model is not None and tokenizer is not None, "Model/tokenizer required for close_embed mode"
        new_number = _find_embed_number(old_number, model, tokenizer, closest=True, rng=rng)
        logger.debug(f"Close embed mode: '{old_number}' -> '{new_number}'")
    elif mode == 'far_embed':
        assert model is not None and tokenizer is not None, "Model/tokenizer required for far_embed mode"
        new_number = _find_embed_number(old_number, model, tokenizer, closest=False, rng=rng)
        logger.debug(f"Far embed mode: '{old_number}' -> '{new_number}'")
    else:
        # Default case for unknown modes
        logger.warning(f"Unknown mode '{mode}', using random mode")
        choices = [x for x in range(-1000, 1000) if x != old_number]
        new_number = rng.choice(choices)
    
    return new_number

def _find_embed_number(
    old_number: float, 
    model, 
    tokenizer, 
    closest: bool = True, 
    rng=None
) -> str:
    """Find number based on embedding similarity using digit embeddings.
    
    Args:
        old_number: The original number to find similar replacement for
        model: The language model for embeddings
        tokenizer: The tokenizer for encoding/decoding
        closest: Whether to find closest (True) or farthest (False) embedding
        rng: Random number generator for tie-breaking
        
    Returns:
        String representation of the replacement number
    """
    import torch
    import re
    
    # Tokenize the number
    number_str = str(old_number)
    token_ids = tokenizer.encode(number_str, add_special_tokens=False)

    # Get embeddings for the tokens
    embeddings = model.get_input_embeddings()
    original_embeds = embeddings(torch.tensor(token_ids, device=model.device))
    
    # Get token IDs for digits 0-9
    digit_tokens = []
    for digit in "0123456789":
        digit_token_ids = tokenizer.encode(digit, add_special_tokens=False)
        if digit_token_ids:  # Take first token if digit is split
            digit_tokens.append(digit_token_ids[0])
    
    # Remove duplicates and convert to tensor
    digit_tokens = list(set(digit_tokens))
    digit_token_tensor = torch.tensor(digit_tokens, device=model.device)
    
    # Get embeddings for digit tokens
    digit_embeds = embeddings(digit_token_tensor)
    
    # Find closest/farthest digit tokens for each original token
    replacement_tokens = []
    for i, orig_embed in enumerate(original_embeds):
        # Check if this token corresponds to a numerical value
        orig_token_id = token_ids[i]
        orig_token_text = tokenizer.decode([orig_token_id], skip_special_tokens=True)
        
        # If not numerical, keep original token and continue
        try:
            float(orig_token_text.strip())
            is_numerical = True
        except ValueError:
            is_numerical = False
            
        if not is_numerical:
            replacement_tokens.append(orig_token_id)
            continue
            
        # Find closest/farthest digit for this numerical token
        distances = torch.norm(digit_embeds - orig_embed.unsqueeze(0), dim=1)
        
        if closest:
            best_idx = torch.argmin(distances)
        else:
            best_idx = torch.argmax(distances)
        
        replacement_tokens.append(digit_tokens[best_idx])
    
    # Decode replacement tokens to text
    replacement_text = tokenizer.decode(replacement_tokens, skip_special_tokens=True)
    return replacement_text

def replace_all_tokens_with_ellipsis(
    text: str, 
    tokenizer=None
) -> Tuple[str, bool]:
    """Replace all tokens in the text with '...' (ellipsis).
    
    Args:
        text: Input text to replace tokens
        tokenizer: Tokenizer to use for tokenization (optional, if None uses simple word splitting)
        
    Returns:
        Tuple of (modified_text, was_modified) - Text with all tokens replaced by '...' 
        and boolean indicating if modification occurred
    """
    if not text or not text.strip():
        logger.debug("Empty or whitespace-only text provided")
        return text, False
    
    try:
        if tokenizer is not None:
            # Use tokenizer for precise token replacement
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) == 0:
                logger.debug("No tokens found in text")
                return text, False
            
            # Replace all tokens with ellipsis
            replacement_text = "..." * len(tokens)
            logger.debug(f"Replaced {len(tokens)} tokens with ellipsis using tokenizer")
            return replacement_text, True
        else:
            # Fallback to simple word-based replacement
            words = text.split()
            if len(words) == 0:
                logger.debug("No words found in text")
                return text, False
            
            # Replace all words with ellipsis, preserving spaces
            replacement_text = " ".join(["..."] * len(words))
            logger.debug(f"Replaced {len(words)} words with ellipsis using word splitting")
            return replacement_text, True
            
    except Exception as e:
        logger.warning(f"Failed to replace tokens with ellipsis: {e}")
        logger.warning("Returning original text")
        return text, False
        

def perturb_chunks_range(
    chunks: list, 
    start_idx: int, 
    end_idx: int, 
    random_seed: Optional[int] = None, 
    mode: str = 'random', 
    delimiter: str = ".",
    replace_all_numbers: bool = True, 
    rephrase_with_gpt: bool = False, 
    remove_boxed_answers_flag: bool = False, 
    model=None, 
    tokenizer=None
) -> Tuple[str, bool, list]:
    """Perturb chunks from start_idx to end_idx and return the combined result.
    
    Args:
        chunks: List of text chunks
        start_idx: Starting index for perturbation (0-based)
        end_idx: Ending index for perturbation (exclusive)
        random_seed: Random seed for reproducible results
        mode: Perturbation mode
        delimiter: Delimiter to use when joining chunks
        replace_all_numbers: Whether to replace all numbers in each chunk
        rephrase_with_gpt: Whether to rephrase chunks with GPT-4
        remove_boxed_answers_flag: Whether to remove boxed answers from perturbed chunks
        model: Model for embedding-based perturbation
        tokenizer: Tokenizer for embedding-based perturbation
        
    Returns:
        Tuple of (perturbed_text, is_perturbed, perturbed_chunks)
    """
    if start_idx == 0:
        return '', False, []
    if start_idx > end_idx or start_idx > len(chunks):
        raise ValueError(f"Checkpoint index {start_idx} is out of range for chunks length {len(chunks)}")

    # Include the current checkpoint
    if start_idx == end_idx:
        chunks_to_perturb = [chunks[start_idx-1]]
    else:
        chunks_to_perturb = chunks[start_idx-1:end_idx-1]  # -1 because when 0, there is no chunk. when 1, it is the first chunk in chunks
    
    perturbed_chunks = []
    any_perturbed = False
    
    for i, chunk in enumerate(chunks_to_perturb):
        if mode == 'replace_all_tokens_ellipsis':
            chunk_perturbed, is_perturbed = replace_all_tokens_with_ellipsis(chunk, tokenizer=tokenizer)
        else:
            chunk_perturbed, is_perturbed = randomly_replace_numbers(
                chunk, random_seed, mode, replace_all_numbers, rephrase_with_gpt, 
                model=model, tokenizer=tokenizer
            )
        if remove_boxed_answers_flag:
            chunk_perturbed = remove_boxed_answers(chunk_perturbed)
        perturbed_chunks.append(chunk_perturbed)
        if i == len(chunks_to_perturb)-1 and is_perturbed:
            any_perturbed = True
            logger.info(f"Perturbed chunk: {chunk}\n -> \n {chunk_perturbed}")
    
    # Combine unperturbed chunks up to current + perturbed chunks from current onward
    chunks_up_to_checkpoint = delimiter.join(chunks[:start_idx-1] + perturbed_chunks)
    #chunks_up_to_checkpoint = remove_boxed_answers(chunks_up_to_checkpoint)
    is_perturbed = any_perturbed
    logger.info(f"Forward perturbation applied: {is_perturbed}")
    return chunks_up_to_checkpoint, is_perturbed, perturbed_chunks

def remove_boxed_answers(text: str) -> str:
    """Remove any appearance of \\boxed{} patterns to prevent answer leakage.
    
    Args:
        text: Input text that may contain \\boxed{} patterns
        
    Returns:
        Text with all \\boxed{} patterns removed
    """
    import re
    # Remove \\boxed{...} patterns (both with and without backslashes)
    patterns = [
        r'\\boxed\{[^}]*\}',  # \\boxed{content}
        r'boxed\{[^}]*\}',    # boxed{content}
        r'\\boxed\{\}',       # \\boxed{}
        r'boxed\{\}',         # boxed{}
    ]
    
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
    # Clean up any extra whitespace that might be left
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def parse_ans(ans: str) -> str:
    """Parse answer from model output.
    
    Args:
        ans: Answer string that may contain boxed format
        
    Returns:
        Parsed answer string
    """
    if isinstance(ans, float):
        return ans
    try:
        ans = ans.split('boxed{')[-1].split('}')[0]
    except Exception:
        pass
    return ans





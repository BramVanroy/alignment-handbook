"""
Language-based sampler for training language models with batches grouped by language.
"""

from typing import Iterator
import torch

from torch.utils.data import Sampler, Dataset
from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_language_grouped_indices(
    dataset: Dataset,
    language_column: str,
    batch_size: int,
    keep_incomplete_batches: bool = False,
    shuffle: bool = True,
    generator=None,
) -> list[int]:
    """
    Create batches of indices grouped by language.
    
    Args:
        dataset: The dataset containing a 'language' column
        batch_size: Size of the batches to create
        keep_incomplete_batches: Whether to keep batches that don't have batch_size samples
        shuffle: Whether to shuffle the indices within each language group and batch order
        generator: Optional random generator for reproducibility
    
    Returns:
        List of indices organized into language-grouped batches
    """
    # Extract language information for all samples
    languages = dataset[language_column]
    
    # Group indices by language
    language_to_indices = {}
    for idx, lang in enumerate(languages):
        if lang not in language_to_indices:
            language_to_indices[lang] = []
        language_to_indices[lang].append(idx)
    
    # Shuffle indices within each language if requested
    if generator is None:
        generator = torch.Generator()
        
    if shuffle:
        for lang in language_to_indices:
            indices = torch.tensor(language_to_indices[lang])
            shuffled_indices = indices[torch.randperm(len(indices), generator=generator)].tolist()
            language_to_indices[lang] = shuffled_indices
    
    # Create batches for each language: this will be a dictionary of language to a list of batches of ints (list of lists of indices)
    lang2batches = {lang: [] for lang in language_to_indices.keys()}
    for lang, indices in language_to_indices.items():
        for start_idx in range(0, len(indices), batch_size):
            batch = indices[start_idx:start_idx + batch_size]
            if len(batch) == batch_size or keep_incomplete_batches:
                lang2batches[lang].append(batch)
    
    # Evenly distribute batches across languages
    batches = list(lang2batches.values())
    flat_indices = evenly_distribute_lists(batches)

    return flat_indices


class LanguageGroupedSampler(Sampler):
    """
    Sampler that groups samples by language to ensure batches contain the same language.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        language_column: str,
        batch_size: int,
        keep_incomplete_batches: bool = False,
        shuffle: bool = True,
        generator=None,
    ):
        """
        Initialize the language-grouped sampler.
        
        Args:
            dataset: Dataset containing a 'language' column
            batch_size: Size of the batches to create
            keep_incomplete_batches: Whether to keep the last batch of each language if incomplete
            shuffle: Whether to shuffle the indices within each language group
            generator: Optional random generator for reproducibility
        """
        self.dataset = dataset
        self.language_column = language_column
        self.batch_size = batch_size
        self.keep_incomplete_batches = keep_incomplete_batches
        self.shuffle = shuffle
        self.generator = generator
        super().__init__(dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator:
        indices = get_language_grouped_indices(
            self.dataset,
            self.language_column,
            self.batch_size,
            self.keep_incomplete_batches,
            shuffle=self.shuffle,
            generator=self.generator,
        )
        return iter(indices)



def evenly_distribute_lists(batches: list[list[list[int]]]) -> list:
    """
    Combine the lists of language batches so that their elements are evenly distributed in the result.
    
    Args:
        lists: A list (languages) of lists (batches) of list (single batch of Ints - sample indices)
        
    Returns:
        A single list with elements from all input sequences evenly distributed
    """
    if not batches:
        return []
        
    # Filter out empty lists
    batches = [lst for lst in batches if lst]
    if not batches:
        return []
    
    # Calculate total length and create result container
    total_length = sum(len(lst) for lst in batches)
    result = [None] * total_length
    
    # Store current positions for each list
    positions = [0] * len(batches)
    
    # Distribute elements
    for position_index in range(total_length):
        # Find which list should contribute next based on distribution fairness
        list_idx = _find_next_list_index(batches, positions, total_length, position_index)
        
        # Add element from selected list
        result[position_index] = batches[list_idx][positions[list_idx]]
        positions[list_idx] += 1
    
    # Flatten sublists
    result = [item for sublist in result for item in sublist]

    return result


def _find_next_list_index(batches: list[list[list[int]]], positions: list[int], total_length: int, current_pos: int) -> int:
    """
    Determine which list should contribute the next element based on fair distribution.
    
    Args:
        lists: List of sequences
        positions: Current position in each list
        total_length: Total number of elements across all lists
        current_pos: Current position in the result list
        
    Returns:
        Index of the list that should contribute the next element
    """
    max_fairness = -1
    selected_idx = -1
    
    for batch_idx, lst in enumerate(batches):
        # Skip if we've used all elements from this list
        if positions[batch_idx] >= len(lst):
            continue
            
        # Calculate how many elements this list has contributed so far
        contributed = positions[batch_idx]
        
        # Calculate how many elements this list should have contributed
        # for perfect distribution at the current position
        target_contribution = (len(lst) / total_length) * (current_pos + 1)
        
        # Fairness measure: how far behind is this list from its fair share
        fairness = target_contribution - contributed
        
        if fairness > max_fairness:
            max_fairness = fairness
            selected_idx = batch_idx
    
    return selected_idx

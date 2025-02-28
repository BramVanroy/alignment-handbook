"""
Custom SFTTrainer with language-based grouped batches.
"""
import torch
import logging
from trl import SFTTrainer

from .sampler import LanguageGroupedSampler

logger = logging.getLogger(__name__)

class LanguageGroupedSFTTrainer(SFTTrainer):
    """
    Supervised Fine-tuning Trainer that creates homogeneous language batches.
    Extends TRL's SFTTrainer with a custom sampler for language-based batch creation.
    """
    def __init__(self, language_column: str, *args, shuffle: bool = True, **kwargs,):
        super().__init__(*args, **kwargs)
        self.language_column = language_column
        self.shuffle = shuffle

    
    def _get_train_sampler(self):
        """
        Override to use LanguageGroupedSampler for training data.
        """
        if self.train_dataset is None:
            return None

        # Check if the dataset has the 'language' column
        if not hasattr(self.train_dataset, "column_names") or self.language_column not in self.train_dataset.column_names:
            logger.warning(
                f"LanguageGroupedSFTTrainer: '{self.language_column}' column not found in dataset. "
                "Falling back to the default sampler."
            )
            # Fall back to the parent class implementation
            return super()._get_train_sampler()
        
        return LanguageGroupedSampler(
            dataset=self.train_dataset,
            language_column=self.language_column,
            batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle,
            generator=torch.Generator().manual_seed(self.args.seed),
        )
    
# data package — place your CSV files here.
# Use dummy_generator.py for offline testing without a real dataset.
from .dummy_generator import make_ratio_dataset, make_structure_dataset

__all__ = ["make_ratio_dataset", "make_structure_dataset"]

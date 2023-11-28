from utils.map_sat_to_lang import SatLanguage
import pickle as pkl
from typing import List
from torch.utils.data import Dataset


class SatSample:
    """Dataset Sample Class"""

    def __init__(self, num_vars:int, num_clauses:int, formula:List[List[int]], is_sat, preferences:str, menu_items: List[str]):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.formula = formula
        self.is_sat = is_sat
        self.preferences = preferences
        self.menu_items = menu_items  # sampled menu_items corresponding to variables


class SatDataset(Dataset):
    def __init__(self, root_path:str, data_path:str):
        self.sat_lang = SatLanguage(root_path)
        self.samples = self.process(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def process(self, data_path:str):
        samples = []
        sat_data = pkl.load(open(data_path, 'rb'))
        for ind, num_vars, num_clauses, formula, is_sat, _ in sat_data: # (5101, 1151), (4721, 1716)
            preferences, menu_items = self.sat_lang.map_2_lang(num_vars, formula, num_clauses)
            samples.append(vars(SatSample(num_vars, num_clauses, formula, is_sat, preferences, menu_items)))
        return samples


# Define a custom collate function that simply passes the data through.
def custom_collate(batch):
    # The batch parameter is a list of whatever your __getitem__ method returns.
    # In this case, we want to bypass the default behavior of converting to tensors,
    # so we will just return the list itself.
    return batch

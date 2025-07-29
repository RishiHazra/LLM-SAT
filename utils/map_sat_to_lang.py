import os
import random
from typing import List, Tuple


class SatLanguage:
    def __init__(self):
        self.food_items = []  # an exhaustive list of food names to sample from
        self.person_names = []  # an exhaustive list of people names to sample from
        self.__init_vars()

    def __init_vars(self):
        self.food_items = open(
            os.path.join(os.environ["DATA_PATH"], "menu_items"), "r"
        ).readlines()
        self.food_items = [item.strip() for item in self.food_items]
        self.person_names = open(
            os.path.join(os.environ["DATA_PATH"], "person_names"), "r"
        ).readlines()
        self.person_names = [item.strip() for item in self.person_names]

    def sample_menu_items(self, num_vars: int) -> List[str]:
        """randomly sample num_vars menu.py items from a list of 50 food items"""
        return random.sample(self.food_items, num_vars)

    def define_constraints(
        self, menu_items: List[str], formula: List[List[int]], num_clauses: int
    ) -> str:
        """define food preferences for num_clause people
        by assigning a one-to-one mapping from formula variables to menu.py items"""
        persons = random.sample(self.person_names, num_clauses)
        all_preferences = ""

        for i in range(num_clauses):
            likes, dislikes = [], []
            for var in formula[i]:
                if var > 0:
                    likes.append(menu_items[var - 1])
                else:
                    dislikes.append(menu_items[-var - 1])
            all_preferences += f"{persons[i]}: "
            if len(likes) > 0:
                all_preferences += f"Likes {', '.join(likes)}. "
            if len(dislikes) > 0:
                all_preferences += f"Dislikes {', '.join(dislikes)}. "
            # all_preferences += f"{persons[i]} prefers either {' or '.join(person_preference)}. "
        return all_preferences

    def map_2_lang(
        self, num_vars: int, formula: List[List[int]], num_clauses: int
    ) -> Tuple[str, List[str]]:
        # for ind, num_vars, num_clauses, formula, is_sat in sat_data:
        menu_items = self.sample_menu_items(num_vars)
        menu_preferences = self.define_constraints(menu_items, formula, num_clauses)
        return menu_preferences, menu_items

    def lang_2_assignment(self, orderable, not_orderable, is_unsat, formula):
        pass

import os
import random
from typing import List, Tuple

class SatLanguage:
    def __init__(self, root_path):
        self.food_items = []  # an exhaustive list of food names to sample from
        self.person_names = []  # an exhaustive list of people names to sample from
        self.__init_vars(root_path)

    def __init_vars(self, root_path: str):
        self.food_items = open(os.path.join(root_path,
                                   'menu_items'), 'r').readlines()
        self.food_items = [item.strip() for item in self.food_items]
        self.person_names = open(os.path.join(root_path,
                                         'person_names'), 'r').readlines()
        self.person_names = [item.strip() for item in self.person_names]

    def sample_menu_items(self, num_vars: int) -> List[str]:
        """randomly sample num_vars menu.py items from a list of 50 food items"""
        return random.sample(self.food_items, num_vars)

    def define_constraints(self, menu_items: List[str], formula: List[List[int]], num_clauses:int) -> str:
        """define food preferences for num_clause people
        by assigning a one-to-one mapping from formula variables to menu.py items"""
        persons = random.sample(self.person_names, num_clauses)
        all_preferences = ''

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


    def map_2_lang(self, num_vars:int, formula:List[List[int]], num_clauses:int) -> Tuple[str, List[str]]:
        # for ind, num_vars, num_clauses, formula, is_sat in sat_data:
        menu_items = self.sample_menu_items(num_vars)
        menu_preferences = self.define_constraints(menu_items, formula, num_clauses)
        # query = f"Objective: Formulate two distinct lists of food items, one denoting what can be ordered (
        # 'Orderable')" \ f" and the other what cannot ('Not Orderable'), to meet the preferences of a group of
        # individuals. " \ f"Each person must find the selection satisfactory based on their likes and dislikes. " \
        # f"If no such combination exists that satisfies all, return ['None']. \nParticipants and Preferences: " \
        # f"{menu_preferences}\n" \ f"Satisfaction Criteria: " \ f"1. A person is satisfied if at least one liked
        # item is 'Orderable' " \ f"OR one disliked item is 'Not Orderable. " \ f"2. No item can appear on both
        # lists. " \ f"3. All participants must be satisfied by the combination of the two lists, or return [
        # 'None'].\n" \ f"Task: Process the above information and generate the 'Orderable' and " \ f"'Not Orderable'
        # as list of strings. " \ f"Ensure that all items are in lowercase. " \ f"If the task is impossible,
        # remember to output ['None']. "\ f"Let's think it step by step." print(query) print('hi')
        return menu_preferences, menu_items

    def lang_2_assignment(self, orderable, not_orderable, is_unsat, formula):
        pass


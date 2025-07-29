import re
from typing import List, Tuple, Dict, Optional
import Levenshtein as lev


def find_best_match(target, string_list):
    # Initialize the best match with the first string in the list and maximum distance
    best_match = string_list[0]
    min_distance = lev.distance(target, best_match)

    # Iterate over the list to find the string with the smallest Levenshtein distance to the target
    for s in string_list[1:]:
        distance = lev.distance(target, s)
        if distance < min_distance:
            min_distance = distance
            best_match = s

    return best_match


def parse_generated_output_sat(raw_output: str) -> Optional[List]:
    # Regex pattern to match the specific format with numbers and boolean values

    pattern = r"<output:\s*\{\s*(?:'?\-?\d+'?:\s*(True|False),?\s*)*\}*>"
    match = re.search(pattern, raw_output)

    # If a match is found, process it
    if match:
        # Find all key-value pairs in the matched string
        kv_pairs = re.findall(r"(-?\d+): (True|False)", match.group(0))
        if len(kv_pairs) == 0:
            kv_pairs = [
                (k.strip("''"), v)
                for k, v in re.findall(r"('?\d+'?): (True|False)", match.group(0))
            ]

        # Process each key-value pair
        result_list = []
        for key, value in kv_pairs:
            # Convert key to int, handling negative keys
            int_key = int(key)
            if int_key < 0:
                int_key = abs(int_key)
                value = "False" if value == "True" else "True"

            # Add the key-value pair to the result dictionary
            if value == "True":
                result_list.append(int_key)

        return result_list

    pattern = r"<output:\s*\{\s*(?:'?\-?\d+'?:\s*(True|False),?\s*)*\}</output>"
    match = re.search(pattern, raw_output)
    # If a match is found, process it
    if match:
        # Find all key-value pairs in the matched string
        kv_pairs = re.findall(r"(-?\d+): (True|False)", match.group(0))
        if len(kv_pairs) == 0:
            kv_pairs = [
                (k.strip("''"), v)
                for k, v in re.findall(r"('?\d+'?): (True|False)", match.group(0))
            ]

        # Process each key-value pair
        result_list = []
        for key, value in kv_pairs:
            # Convert key to int, handling negative keys
            int_key = int(key)
            if int_key < 0:
                int_key = abs(int_key)
                value = "False" if value == "True" else "True"

            # Add the key-value pair to the result dictionary
            if value == "True":
                result_list.append(int_key)

        return result_list

    pattern = r"```python\{\s*(?:'?\-?\d+'?:\s*(True|False),?\s*)*\}```"
    match = re.search(pattern, raw_output)
    # If a match is found, process it
    if match:
        # Find all key-value pairs in the matched string
        kv_pairs = re.findall(r"(-?\d+): (True|False)", match.group(0))
        if len(kv_pairs) == 0:
            kv_pairs = [
                (k.strip("''"), v)
                for k, v in re.findall(r"('?\d+'?): (True|False)", match.group(0))
            ]

        # Process each key-value pair
        result_list = []
        for key, value in kv_pairs:
            # Convert key to int, handling negative keys
            int_key = int(key)
            if int_key < 0:
                int_key = abs(int_key)
                value = "False" if value == "True" else "True"

            # Add the key-value pair to the result dictionary
            if value == "True":
                result_list.append(int_key)

        return result_list

    matches = re.findall(r"```python*\s*(.*?)\s*```", raw_output, re.DOTALL)
    # If there are matches, return the last one
    if matches:
        if matches[-1] == "{}":
            return []
    # If no match is found, return an empty dictionary
    return None


def parse_generated_output(
    raw_output: str,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    # Regular expression to find the lists
    pattern1 = r"```python\s+orderable\s*=\s*(\[[^\]]*\])\s*(,|\n)\s*not_orderable\s*=\s*(\[[^\]]*\])\s*```"
    match1 = re.search(pattern1, raw_output)

    pattern2 = (
        r"orderable\s*=\s*(\[[^\]]*\])\s*(,|\n)\s*not_orderable\s*=\s*(\[[^\]]*\])\s*"
    )
    match2 = re.search(pattern2, raw_output)

    pattern3 = r"(?i)(orderable\s*[:=]\s*(\[[^\]]*\]|\S+(?:,\s*\S+)*))\s*(?:,|\n)?\s*(not[_ ]orderable)\s*[:=]\s*(\[[^\]]*\]|\S+(?:,\s*\S+)*)"
    match3 = [match for match in re.finditer(pattern3, raw_output, re.DOTALL)]

    pattern4 = r"(?i)(orderable\s*[:=]\s*(\[[^\]]*\]|\S+(?:,\s*\S+)*))|((not[_ ]orderable)\s*[:=]\s*(\[[^\]]*\]|\S+(?:,\s*\S+)*))"
    match4 = [match for match in re.finditer(pattern4, raw_output, re.DOTALL)]

    # Extracting the lists if found
    if match1:
        try:
            orderable_list = eval(match1.group(1))
        except NameError:
            orderable_list = match1.group(1).strip("[]").split(",")
        except SyntaxError:
            orderable_list = match1.group(1).strip("[]").split(",")
            orderable_list = [item.strip().strip("''") for item in orderable_list]
        try:
            not_orderable_list = eval(match1.group(3))
        except NameError:
            not_orderable_list = match1.group(3).strip("[]").split(",")
        except SyntaxError:
            not_orderable_list = match1.group(3).strip("[]").split(",")
            not_orderable_list = [
                item.strip().strip("''") for item in not_orderable_list
            ]
    elif match2:
        final_lists = re.findall(pattern2, raw_output)[-1]
        orderable_list, not_orderable_list = final_lists[0], final_lists[-1]
        if orderable_list == "[]":
            orderable_list = []
        else:
            orderable_list = orderable_list.strip("[]").split(",")
        if not_orderable_list == "[]":
            not_orderable_list = []
        else:
            not_orderable_list = not_orderable_list.strip("[]").split(",")
    elif match3:
        match = match3[-1]
        orderable_str = match.group(2).strip("[]")
        not_orderable_str = match.group(4).strip("[]")
        orderable_list = [item.strip(" '\"") for item in orderable_str.split(",")]
        not_orderable_list = [
            item.strip(" '\"") for item in not_orderable_str.split(",")
        ]
    elif match4:
        match = match4[-1]
        orderable_str = match.group(2).strip("[]")
        not_orderable_str = match.group(4).strip("[]")
        orderable_list = [item.strip(" '\"") for item in orderable_str.split(",")]
        not_orderable_list = [
            item.strip(" '\"") for item in not_orderable_str.split(",")
        ]
    else:
        orderable_list = not_orderable_list = None
        return orderable_list, not_orderable_list

    # postprocess
    orderable_list = [
        item.strip().strip("''").strip('""').lower() for item in orderable_list
    ]
    not_orderable_list = [
        item.strip().strip("''").strip('""').lower() for item in not_orderable_list
    ]
    return orderable_list, not_orderable_list


# Further refined function to correctly parse the CNF formula from LaTeX format
def parse_generated_output_translate(
    raw_output: str, item_to_number: Dict[str, int], menu_items: List
) -> List[List]:
    # Extract the clauses from the LaTeX formatted string
    # Remove all LaTeX formatting and split by 'and' (\\land)
    raw_output = (
        raw_output.replace("lnot", "neg")
        .replace("wedge", "land")
        .replace("vee", "lor")
        .replace("\\text{", "")
        .replace("}", "")
        .replace(r"\\", "")
    )
    raw_output = raw_output.replace("\\\\\n&", "").replace("&", "")
    clauses = re.findall(r"\((.*?)\)", raw_output)

    parsed_clauses = []
    for clause in clauses:
        clause = clause.replace("(", "").replace(")", "")
        # Split each clause into literals separated by 'or' (\\lor)
        literals = clause.split("\\lor")
        parsed_clause = []
        for literal in literals:
            literal = literal.strip()
            # Check if the literal is negated
            if literal.startswith("\\neg"):
                item = literal[4:].strip()  # Remove '\\neg ' prefix
                # item = find_best_match(item, sample_dict['menu_items'])
                parsed_clause.append(-item_to_number[item.lower()])
            else:
                item = literal.lower()
                item = find_best_match(literal.lower(), menu_items)
                parsed_clause.append(item_to_number[item])
        parsed_clauses.append(parsed_clause)

    return parsed_clauses

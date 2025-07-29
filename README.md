# Have Large Language Models Learned to Reason? A Characterization via 3-SAT Phase Transitions
This repository contains the code and data for our **COLM 2025** paper [Have Large Language Models Learned to Reason?
A Characterization via 3-SAT](https://arxiv.org/pdf/2504.03930).

## SAT Solving and Model Counting

In `src/` the files `sat_solve.py` and `model_counting.py` implement, respectively, the sat solving module and the model counter module.

## Dataset Generation

- `data_maker.py` is a dataset generator
  - 'DataMakerSR` class can generate the SR(n) family of formulas, described at the beginning of Section 4 in [this paper](https://arxiv.org/pdf/1802.03685.pdf)
- `data_loader.py` loads the dataset and provides some stats for it
- `scripts/` contains useful scripts to generate datasets and run experiments

## Dependencies

### Python

- Python (>= 3.9)
- numpy (>= 1.24)
- [PyMiniSolvers](https://github.com/liffiton/PyMiniSolvers) (installable through `/setup.sh`)
- [d4](https://github.com/crillab/d4) (installable through `/setup.sh`)
- matplotlib (>= 3.7.1)
- pandas (>= 2.0.2)

### Boost Multiprecision library

This library is necessary to compile `d4`. You can install it in the following way.

#### Ubuntu/Debian

```sudo apt-get install libboost-all-dev```

#### macOS (using Homebrew)

```brew install boost```

## Plotting

The folder `plots/` contains the scripts used to generate plots.

## Getting Started (SAT Solver)

1. Run `/setup.sh` and `pip install -r requirements.txt`
2. You can either generate a dataset or use an already existing one
  
    2a. To generate one see `gen_data.sh` and `data_maker.py`

    2b. To use an already existing one just copy paste it in the `/data/` folder

3. (Optional) Run `data_loader.py` passing the path to your dataset to obtain a brief analysis of your dataset

## Getting Started (LLMs)

1. Run `/setup.sh` and `pip install -r requirements.txt`
2. We provide the datasets we used for the submission.
    - The file `dataset.pkl` contains all the formulas with integer alpha values.
    - The file `dataset_float_alpha.pkl` contains all the formulas with float alpha values.
3. The file `main.py` is used to run the experiments with different LLMs.
4. You would need a key for each. For e.g., for GPT-4o the [OPENAI_API_KEY](https://openai.com/index/openai-api/)

```python main.py --ablation <ablation type> --model_name <model name> --sat_class <sat complexity class>```

Here, 
* ablation_type = ["menu", "sat", "translate"], "menu" is for SAT-Menu, "sat" is for SAT-CNF
* model_name = ["deepseek_v3", "gemini_25_pro", "deepseek", "gemini_20", "gpt-4o", "claude_37_sonnet"],
* sat_class = ["2sat", "1_3_horn_sat", "1_2_horn_sat", "3sat_float", "3sat_int"]. 3-SAT data is available with float and int alpha values. Our paper results report a combination of the two.
 ---
The system prompts can be found in `system_messages/`  

The raw outputs and annotations can be found in `extras/` folder.


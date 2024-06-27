# LLM-SAT

Code and data for our paper 
## Can Large Language Models Reason? A Characterization via 3-SAT Phase Transitions

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

Necessary to compile `d4`. You can install it in the following way.

#### Ubuntu/Debian

```sudo apt-get install libboost-all-dev```

#### macOS (using Homebrew)

```brew install boost```

## Plotting

The folder `plots/` contains the scripts used to generate plots.

## Getting Started

1. Run `/setup.sh` and `pip install -r requirements.txt`
2. You can either generate a dataset or use an already existing one
  
    2a. To generate one see `gen_data.sh` and `data_maker.py`

    2b. To use an already existing one just copy paste it in the `/data/` folder

3. (Optional) Run `data_loader.py` passing the path to your dataset to obtain a brief analysis of your dataset

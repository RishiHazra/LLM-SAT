# LLM-SAT

Need to describe the LLM part.

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
3. The file `main.py` is used to run the experiments with GPT, LLama, and Mixtral.
4. To run GPT-* models, you would need a [OPENAI_API_KEY](https://openai.com/index/openai-api/)
5. To run Llama models, apply for a license through this [link](https://huggingface.co/meta-llama). Once you have the license, create an account on Huggingface and update your Llama token through this [link](https://huggingface.co/meta-llama). Set up your [HF_ACCESS_TOKEN](https://huggingface.co/docs/hub/en/security-tokens).
6. For PaLM and Gemini Pro, you would need to set up an account on Google Cloud. Check the Vertex AI setup [here](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini).


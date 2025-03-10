# DEVELOPING FICTITIOUS COUNTRY MAPS THROUGH GENERATIVE AI TECHNIQUES

## Overview
This project is part of my master's thesis, **"Developing Fictitious Country Maps Through Generative AI Techniques"**. The goal is to generate realistic, AI-generated maps using diffusion models trained on satellite imagery. This thesis was done in a collaboration with the United Nations.

## File Descriptions
- **calc_fid.py** - Computes the Frechet Inception Distance (FID) to evaluate the quality of generated images.
- **compute_statistics.py** - Computes training dataset statistics.
- **configs.py** - Stores configuration settings and hyperparameters for the model.
- **create_map.py** - Generates fictitious maps using the trained diffusion model.
- **datasets.py** - Handles data loading, preprocessing, and augmentation.
- **do_a_sample.py** - Runs a sample inference using the trained model.
- **environment.yml** - Specifies the Conda environment setup with required dependencies.
- **fp16_util.py** - Utilities for mixed precision (FP16) training to optimize performance.
- **main.py** - The main script to train the diffusion model and generate samples.
- **main_yaml.py** - Alternative training script that loads configurations from a YAML file, used for unconditional sampling.
- **misc.py** - Utility functions on tensors used across the project.
- **models.py** - Defines the architecture of the diffusion model.
- **nn.py** - Contains neural network utility functions.
- **trainer.py** - Implements the training loop and model optimization.
- **unetG.py** - Defines the U-Net generator used in the diffusion model.

## Installation
### Prerequisites
- Python 3.11+
- Conda (or Miniconda/Anaconda)
- CUDA (if training on GPU)

### Setting Up the Environment
The environment can be set up directly using the provided `environment.yml` file:
```sh
conda env create -f environment.yml
conda activate my-environment
```

## Usage
### Help
To view available options:
```sh
python main.py --help
```

### Training
Set configurations in `train_diffusion()` in `main.py` and then run:
```sh
python main.py diffusion
```
Results are saved in `./projects/`.

### Generating Fictitious Maps
After training, generate maps:
```sh
python create_map.py 
```

## Contributions
This work is inspired by OpenAI's diffusion model and Sentinel-2 satellite imagery processing.

## Author
**Aleksandra Jastrzębska** - Master in **Geospatial Technologies** (Erasmus Mundus)

## License
This repository is open-source under the MIT License.

## Acknowledgments
- OpenAI's **Guided Diffusion** repository
- Sentinel-2 satellite data sources
- Repository of Leon Pielage 


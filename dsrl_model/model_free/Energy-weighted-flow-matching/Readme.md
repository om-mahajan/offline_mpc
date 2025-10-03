# Implementation of Energy-weighted Flow Matching for offline RL

> Some of the code implementation is from  [Contrastive Energy Prediction](https://github.com/ChenDRAG/CEP-energy-guided-diffusion)

### Requirements

Install `mujoco210` in `~/.mujoco/mujoco210` install the conda environment
```[bash]
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/bin/nvidia
conda env create -f environment.yml
```

## Toy2D experiments

For reproducing toy 2D experiments in the paper using the energy-weighted flow matching
```.bash
python -u bandit_toy.py --expid EXPID --env moons --diffusion_steps 15 --seed 0 --alpha 3 --schedule OT # Flow matching
python -u bandit_toy.py --expid EXPID --env moons --diffusion_steps 15 --seed 0 --alpha 3 --schedule linear # diffusion

```

And the data is visualized via `draw_toy.ipynb`

For reproducing toy 2D experiments in the paper using the classifier-free guidance, run
```.bash
python -u cfg.py --expid EXPID --env moons --diffusion_steps 15 --seed 0 --alpha 3
```
Checkpoints will be stored in the `./models` folder. Visualization scripts are provided in `draw_toy.ipynb`.

## Offline RL datasets

For reproducing the result in D4RL benchmarks using energy-weighted flow matching or diffusion model, run
```[bash]
python -u train_rl.py --expid EXPID --seed 0 --env walker2d-medium-expert-v2 --schedule OT # Flow matching
python -u train_rl.py --expid EXPID --seed 0 --env walker2d-medium-expert-v2 --schedule linear # diffusion
```

# On the Robustness of Context- and Gradient-based Meta-Reinforcement Learning Algorithms

For installing MuJoCo refer [here](https://github.com/openai/mujoco-py).


## Setting the environment

```bash
virtualenv venv --python=python3.7
source venv/bin/activate
pip install -r requirements.txt
```

## Downloading the data

We provide the [link](https://polybox.ethz.ch/index.php/s/orR8QC1lON12S5K) for the data necessary for evaluating how good MACAW performs in regard to out-of-distribution testing for 90° and the Bernoulli-Bandit experiment. Please download the folders and place them in `./data/{bandit/ant_dir_1}`.

Due to the large size of data, other files are available upon request.

## Reproduce MACAW results

Run offline meta-training with periodic online evaluations with any of the following commands e.g.
    
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit3 --task_config config/bandit/3tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit5 --task_config config/bandit/5tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit10 --task_config config/bandit/10tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit20 --task_config config/bandit/20tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit35 --task_config config/bandit/35tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    
    python run.py --device cuda:0 --name macaw_ant --log_dir log/ant_1 --task_config config/ant_dir/50tasks_offline_1.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json
    
Outputs (tensorboard logs) will be written to the `log/` directory.


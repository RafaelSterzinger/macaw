## Installing the environment

    # Install Python 3.7.9 if necessary
    $ pyenv install 3.7.9
    $ pyenv shell 3.7.9
    
    $ python --version
    Python 3.7.9
    
    $ python -m venv env
    $ source env/bin/activate
    $ pip install -r requirements.txt

## Downloading the data

Will give polybox link

The offline data used for MACAW can be found [here](https://drive.google.com/drive/folders/1kJEAYNWBYRD4ZIE3Ww0epXGM2VGelrQC?usp=sharing). Download it and use the default name (`macaw_offline_data`) for the folder where the four data directories are stored. [gDrive](https://github.com/prasmussen/gdrive) might be useful here if downloading from the Google Drive GUI is not an option.

## Running MACAW 🦜

Run offline meta-training with periodic online evaluations with any of the scripts in `scripts/`. e.g.
    
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit3 --task_config config/bandit/3tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit5 --task_config config/bandit/5tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit10 --task_config config/bandit/10tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit20 --task_config config/bandit/20tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    python run.py --device cuda:0 --name macaw_bandit --log_dir log/bandit35 --task_config config/bandit/35tasks_offline.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json 
    
    python run.py --device cuda:0 --name macaw_ant --log_dir log/ant_1 --task_config config/ant_dir/50tasks_offline_1.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json
    python run.py --device cuda:0 --name macaw_ant --log_dir log/ant_2 --task_config config/ant_dir/50tasks_offline_2.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json
    python run.py --device cuda:0 --name macaw_ant --log_dir log/ant_3 --task_config config/ant_dir/50tasks_offline_3.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json
    python run.py --device cuda:0 --name macaw_ant --log_dir log/ant_4 --task_config config/ant_dir/50tasks_offline_4.json --macaw_params config/alg/standard.json --macaw_override_params config/alg/overrides/no_override.json

    ...
    
Outputs (tensorboard logs) will be written to the `log/` directory.

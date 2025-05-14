to clone the repository use:
git clone --recurse-submodules git@github.com:paulutsch/pm-25.git
this command will:
clone the main repository (pm-25)
also automatically initialize and clone all submodules (lm-evaluation-harness)

install dependencies of lm-evaluation-harness:
cd lm-evaluation-harness
pip install -e .

install requirements.txt:
cd ..
pip install -r requirements.txt

export your wandb key:
export WANDB_API_KEY=your key
Note: to find you key go to your wandb account to https://wandb.ai/authorize

to start sft tuning:
- set parameters in config.yaml
Note: the first tuning parameter must be set as sft
Note2: set base_model you want to tune and sft_dataset for tuning
- python tuning.py --config config.yaml

to start dpo tuning:
- set parameters in config.yaml
Note: the first tuning parameter must be set as dpo
Note2: set base_model (it will function as a reference model during dpo-tuning), sft_model you want to tune (the size of base_model and sft_model should match, for example, 1B in both cases) and preference_dataset for tuninng
- python tuning.py --config config.yaml

after tuning you will get a directory there you will find:
- the tunde model
- the checkpoints of tuning (see directory outputs)
- the config you used (see the file used_config)
- wandb logs



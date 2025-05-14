**to clone the repository use**
```bash
git clone --recurse-submodules git@github.com:paulutsch/pm-25.git
```
*this command will*:
1. clone the main repository (pm-25)
2. also automatically initialize and clone all submodules (lm-evaluation-harness)

**install dependencies of lm-evaluation-harness**:
```bash
cd lm-evaluation-harness
pip install -e .
```

**install requirements.txt**:
```bash
cd ..
pip install -r requirements.txt
```

**export your wandb key**:
```bash
export WANDB_API_KEY=your key
```
*Note: to find you key go to your wandb account to https://wandb.ai/authorize*

**to start sft tuning**:
1. set parameters in config.yaml
*Note: the first tuning parameter must be set as sft*
*Note2: set base_model you want to tune and sft_dataset for tuning*
2. run
```bash
python tuning.py --config config.yaml
```

**to start dpo tuning**:
1. set parameters in config.yaml
*Note: the first tuning parameter must be set as dpo*
*Note2: set base_model (it will function as a reference model during dpo-tuning), sft_model you want to tune (the size of base_model and sft_model should match, for example, 1B in both cases) and preference_dataset for tuninng*
2. run
```bash
python tuning.py --config config.yaml
```

**after tuning you will get a directory there you will find**:
1. the tuned model
2. the checkpoints of tuning (see directory outputs)
3. the config you used (see the file used_config)
4. wandb logs



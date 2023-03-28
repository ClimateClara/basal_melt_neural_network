
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/run_training_whole_dataset.py xsmall96 extrap std newbasic2 07 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/experiments/xsmall96_extrap_std_wholedataset_newbasic2_07.log

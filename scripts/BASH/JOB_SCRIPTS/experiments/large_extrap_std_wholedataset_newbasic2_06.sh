
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/run_training_whole_dataset.py large extrap std newbasic2 06 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/experiments/large_extrap_std_wholedataset_newbasic2_06.log

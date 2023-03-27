
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/run_training_whole_dataset.py small extrap std newbasic2 03 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/experiments/small_extrap_std_wholedataset_newbasic2_03.log

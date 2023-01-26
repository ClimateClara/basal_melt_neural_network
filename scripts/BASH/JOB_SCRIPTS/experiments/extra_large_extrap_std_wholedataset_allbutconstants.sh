
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/run_training_whole_dataset.py extra_large extrap std allbutconstants 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/experiments/extra_large_extrap_std_wholedataset_allbutconstants.log

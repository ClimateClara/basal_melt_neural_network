
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/run_cross_validation_NN_experiments.py medium 00 42 extrap std onlyTSisfdraft 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/experiments/medium_extrap_std_noisf42_notblock00_onlyTSisfdraft.log

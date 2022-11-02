
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/run_cross_validation_NN_experiments.py medium 05 00 extrap std TSdraftbotandiceddandwcdreldGL 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/experiments/medium_extrap_std_noisf00_notblock05_TSdraftbotandiceddandwcdreldGL.log

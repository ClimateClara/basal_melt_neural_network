
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/compute_1D_evalmetrics_shuffling_deepensemble_Smith.py large extrap std newbasic2 bi646 S_mean 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/apply_2D_and_1D/shuffling_S_mean_ensmean_large_extrap_std_newbasic2_bi646.log

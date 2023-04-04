
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/compute_1D_evalmetrics_shuffling_deepensemble_Smith.py large extrap_shuffboth std newbasic2 bf663 slope_bed_lat 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/apply_2D_and_1D/shuffling_slope_bed_lat_ensmean_large_extrap_shuffboth_std_newbasic2_bf663.log

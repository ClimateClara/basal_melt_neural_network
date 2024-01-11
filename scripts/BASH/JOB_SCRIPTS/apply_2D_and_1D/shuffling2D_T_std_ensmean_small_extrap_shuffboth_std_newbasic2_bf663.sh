
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/compute_2D_evalmetrics_shuffling_deepensemble_Smith.py small extrap_shuffboth std newbasic2 bf663 T_std 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/apply_2D_and_1D/shuffling2D_T_std_ensmean_small_extrap_shuffboth_std_newbasic2_bf663.log

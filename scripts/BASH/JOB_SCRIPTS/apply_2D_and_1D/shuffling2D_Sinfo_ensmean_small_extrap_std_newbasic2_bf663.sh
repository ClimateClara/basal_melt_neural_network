
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/compute_2D_evalmetrics_shuffling_deepensemble_Smith.py small extrap std newbasic2 bf663 Sinfo 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/apply_2D_and_1D/shuffling2D_Sinfo_ensmean_small_extrap_std_newbasic2_bf663.log

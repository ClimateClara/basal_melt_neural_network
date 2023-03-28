
#!/bin/bash

conda activate neuralnet
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/compute_2D_deepensemble_Smith.py xsmall96 extrap std newbasic2 bi646 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/apply_2D_and_1D/apply2D_ensmean_xsmall96_extrap_std_newbasic2_bi646.log

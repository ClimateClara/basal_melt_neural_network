
#!/bin/bash

conda activate py38
python -u /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON/run_cdo_interp.py 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS//OPM031_interp_suite.log


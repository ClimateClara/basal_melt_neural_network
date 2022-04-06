
#!/bin/bash

conda activate py38
python -u /bettik/burgardc/SCRIPTS/basal_melt_param/scripts/PYTHON/run_cdo_interp.py 2>&1 | tee /bettik/burgardc/SCRIPTS/basal_melt_param/scripts/BASH/JOB_SCRIPTS//OPM016_interp.log


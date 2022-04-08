#!/bin/bash

path_jobscripts=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/
path_outfiles=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/
path_python=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON
path_jobid=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/JOB_STD_OUTPUT

nemo_run=OPM031

i=interp_suite

cat << EOF > $path_jobscripts/${nemo_run}_${i}.sh

#!/bin/bash

conda activate py38
python -u $path_python/run_cdo_interp.py 2>&1 | tee $path_outfiles/${nemo_run}_${i}.log

EOF
chmod +x $path_jobscripts/${nemo_run}_${i}.sh


oarsub -S -n ${nemo_run}_${i} --stdout $path_jobid/${nemo_run}_${i}.o%jobid%  --stderr $path_jobid/${nemo_run}_${i}.e%jobid% -l nodes=1/core=4,walltime=10:00:00 --project ice_speed -p "network_address='luke62'" $path_jobscripts/${nemo_run}_${i}.sh

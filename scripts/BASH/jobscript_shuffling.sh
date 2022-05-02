#!/bin/bash

path_jobscripts=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/
path_outfiles=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/
path_python=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON
path_jobid=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/JOB_STD_OUTPUT

nemo_run=OPM021

nn_model=20220427-1051

cat << EOF > $path_jobscripts/${nemo_run}_${nn_model}_shuffle.sh

#!/bin/bash

conda activate neuralnet
python -u $path_python/post_processing/shuffle_vars_job.py 2>&1 | tee $path_outfiles/${nemo_run}_${nn_model}_shuffle.log

EOF
chmod +x $path_jobscripts/${nemo_run}_${nn_model}_shuffle.sh


oarsub -S -n ${nemo_run}_${nn_model}_shuffle --stdout $path_jobid/${nemo_run}_${nn_model}_shuffle.o%jobid%  --stderr $path_jobid/${nemo_run}_${nn_model}_shuffle.e%jobid% -l nodes=1/core=8,walltime=10:00:00 --project ice_speed -p "network_address='luke62'" $path_jobscripts/${nemo_run}_${nn_model}_shuffle.sh
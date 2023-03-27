#!/bin/bash

## RUN CROSS VALIDATION JOBS

mod_size=large  #'mini', 'small', 'medium', 'large', 'extra_large'
TS_opt=extrap # extrap, whole, thermocline
norm_method=std # std, interquart, minmax
exp_name=newbasic2
seed_nb=01


path_jobscripts=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/experiments
path_outfiles=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/experiments

path_python=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON
path_jobid=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/JOB_STD_OUTPUT

cat <<EOF > $path_jobscripts/${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}.sh 

#!/bin/bash

conda activate neuralnet
python -u $path_python/run_training_whole_dataset.py ${mod_size} ${TS_opt} ${norm_method} ${exp_name} ${seed_nb} 2>&1 | tee $path_outfiles/${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}.log
EOF

chmod +x $path_jobscripts/${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}.sh

oarsub -S -n ${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb} --stdout $path_jobid/${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}.o%jobid%  --stderr $path_jobid/${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}.e%jobid% -l nodes=1/core=4,walltime=05:00:00 --project mais -p "network_address='luke62'" $path_jobscripts/${mod_size}_${TS_opt}_${norm_method}_wholedataset_${exp_name}_${seed_nb}.sh

# to remove if no CV!!!






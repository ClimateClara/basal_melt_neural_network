#!/bin/bash

## RUN CROSS VALIDATION JOBS

mod_size=xsmall96  #'mini', 'small', 'medium', 'large', 'extra_large'
TS_opt=extrap # extrap, whole, thermocline
norm_method=std # std, interquart, minmax
exp_name=newbasic2
nemo_run=bf663


path_jobscripts=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/apply_2D_and_1D
path_outfiles=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/apply_2D_and_1D

path_python=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON
path_jobid=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/JOB_STD_OUTPUT

cat <<EOF > $path_jobscripts/apply2D_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.sh 

#!/bin/bash

conda activate neuralnet
python -u $path_python/compute_2D_deepensemble_Smith.py ${mod_size} ${TS_opt} ${norm_method} ${exp_name} ${nemo_run} 2>&1 | tee $path_outfiles/apply2D_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.log
EOF

chmod +x $path_jobscripts/apply2D_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.sh

oarsub -S -n apply2D_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run} --stdout $path_jobid/apply2D_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.o%jobid%  --stderr $path_jobid/apply2D_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.e%jobid% -l nodes=1/core=4,walltime=04:00:00 --project mais -p "network_address='luke62'" $path_jobscripts/apply2D_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.sh

# to remove if no CV!!!









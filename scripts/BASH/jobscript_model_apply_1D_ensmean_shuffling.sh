#!/bin/bash

## RUN CROSS VALIDATION JOBS

mod_size=large #'mini', 'small', 'medium', 'large', 'extra_large'
TS_opt=extrap # extrap, whole, thermocline #extrap_shuffboth
norm_method=std # std, interquart, minmax
exp_name=newbasic2
nemo_run=bf663

for vv in dGL dIF corrected_isfdraft bathy_metry slope_bed_lon slope_bed_lat slope_ice_lon slope_ice_lat theta_in salinity_in T_mean S_mean T_std S_std watercolumn position slopesbed slopesice Tinfo Sinfo
do 


path_jobscripts=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/apply_2D_and_1D
path_outfiles=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/apply_2D_and_1D

path_python=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON
path_jobid=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/JOB_STD_OUTPUT

cat <<EOF > $path_jobscripts/shuffling_${vv}_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.sh 

#!/bin/bash

conda activate neuralnet
python -u $path_python/compute_1D_evalmetrics_shuffling_deepensemble_Smith.py ${mod_size} ${TS_opt} ${norm_method} ${exp_name} ${nemo_run} ${vv} 2>&1 | tee $path_outfiles/shuffling_${vv}_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.log
EOF

chmod +x $path_jobscripts/shuffling_${vv}_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.sh

oarsub -S -n shuffling_${vv}_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run} --stdout $path_jobid/shuffling_${vv}_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.o%jobid%  --stderr $path_jobid/shuffling_${vv}_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.e%jobid% -l nodes=1/core=3,walltime=08:00:00 --project mais -p "network_address='luke62'" $path_jobscripts/shuffling_${vv}_ensmean_${mod_size}_${TS_opt}_${norm_method}_${exp_name}_${nemo_run}.sh

# to remove if no CV!!!

done










#!/bin/bash

## RUN CROSS VALIDATION JOBS

layer_nb=5 #'mini', 'small', 'medium', 'large', 'extra_large' # still need L and XL for both CV
layer_size=96
TS_opt=extrap # extrap, whole, thermocline
norm_method=std # std, interquart, minmax
exp_name=newbasic


for ii in 10 11 12 13 18 22 23 24 25 30 31 33 38 39 40 42 43 44 45 47 48 51 52 53 54 55 58 61 65 66 69 70 71 73 75 
#for ii in 73 75
do
isf_out=$ii
#isf_out=00

#for tt in {01..13} # restart at 03 for mini # 13 need to be redone as well
#do
#tblock_out=$tt
tblock_out=00

path_jobscripts=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/experiments
path_outfiles=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/experiments

path_python=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON
path_jobid=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/JOB_STD_OUTPUT

cat <<EOF > $path_jobscripts/${layer_nb}_${layer_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}_${exp_name}.sh 

#!/bin/bash

conda activate neuralnet
python -u $path_python/run_cross_validation_NN_experiments_gridsearch.py ${layer_nb} ${layer_size} ${tblock_out} ${isf_out} ${TS_opt} ${norm_method} ${exp_name} 2>&1 | tee $path_outfiles/${layer_nb}_${layer_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}_${exp_name}.log
EOF

chmod +x $path_jobscripts/${layer_nb}_${layer_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}_${exp_name}.sh

oarsub -S -n ${layer_nb}_${layer_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}_${exp_name} --stdout $path_jobid/${layer_nb}_${layer_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}_${exp_name}.o%jobid%  --stderr $path_jobid/${layer_nb}_${layer_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}_${exp_name}.e%jobid% -l nodes=1/core=4,walltime=10:00:00 --project mais -p "network_address='luke62'" $path_jobscripts/${layer_nb}_${layer_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}_${exp_name}.sh

# to remove if no CV!!!
done






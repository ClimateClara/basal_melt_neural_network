#!/bin/bash

## RUN CROSS VALIDATION JOBS

mod_size=medium  #'mini', 'small', 'medium', 'large', 'extra_large'
TS_opt=extrap # extrap, whole, thermocline
norm_method=std # std, interquart, minmax


for ii in 10 11 12 13 18 22 23 24 25 30 31 33 38 39 40 42 43 44 45 47 48 51 52 53 54 55 58 61 65 66 69 70 71 73 75 # a partir de 31
#for ii in 31 33 38 39 40 42 43 44 45 47 48 51 52 53 54 55 58 61 65 66 69 70 71 73 75
do
isf_out=$ii
#isf_out=00

#for tt in {01..13}
#do
#tblock_out=$tt
tblock_out=00

path_jobscripts=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/test_wo_isf_constants
path_outfiles=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/test_wo_isf_constants

path_python=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON
path_jobid=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/JOB_STD_OUTPUT

cat <<EOF > $path_jobscripts/${mod_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}.sh 

#!/bin/bash

conda activate neuralnet
python -u $path_python/run_cross_validation_NN.py ${mod_size} ${tblock_out} ${isf_out} ${TS_opt} ${norm_method} 2>&1 | tee $path_outfiles/${mod_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}.log
EOF

chmod +x $path_jobscripts/${mod_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}.sh

oarsub -S -n ${mod_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out} --stdout $path_jobid/${mod_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}.o%jobid%  --stderr $path_jobid/${mod_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}.e%jobid% -l nodes=1/core=6,walltime=10:00:00 --project mais -p "network_address='luke62'" $path_jobscripts/${mod_size}_${TS_opt}_${norm_method}_noisf${isf_out}_notblock${tblock_out}.sh

# to remove if no CV!!!
done






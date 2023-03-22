#!/bin/bash

## PREPARE INPUT DATA PARALLELLY

TS_opt=extrap # extrap, whole, thermocline


for ii in 10 11 12 13 18 22 23 24 25 30 31 33 38 39 40 42 43 44 45 47 48 51 52 53 54 55 58 61 65 66 69 70 71 73 75 # do again from 48 on
#for ii in 31 33 38 39 40 42 43 44 45 47 48 51 52 53 54 55 58 61 65 66 69 70 71 73 75
do
isf_out=$ii
#isf_out=00

#for tt in {03..13} # restart at 03 for mini # from 8 need to be redone as well
#do
#tblock_out=$tt
tblock_out=00

path_jobscripts=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_SCRIPTS/prep_indata
path_outfiles=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/BASH/JOB_OUTFILES/prep_indata

path_python=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/PYTHON
path_jobid=/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/scripts/JOB_STD_OUTPUT

cat <<EOF > $path_jobscripts/${TS_opt}_noisf${isf_out}_notblock${tblock_out}.sh 

#!/bin/bash

conda activate neuralnet
python -u $path_python/prepare_indata_parallely.py ${tblock_out} ${isf_out} ${TS_opt}  2>&1 | tee $path_outfiles/${TS_opt}_noisf${isf_out}_notblock${tblock_out}.log
EOF

chmod +x $path_jobscripts/${TS_opt}_noisf${isf_out}_notblock${tblock_out}.sh

oarsub -S -n ${TS_opt}_noisf${isf_out}_notblock${tblock_out} --stdout $path_jobid/${TS_opt}_noisf${isf_out}_notblock${tblock_out}.o%jobid%  --stderr $path_jobid/${TS_opt}_noisf${isf_out}_notblock${tblock_out}.e%jobid% -l nodes=1/core=2,walltime=05:00:00 --project mais -p "network_address='luke62'" $path_jobscripts/${TS_opt}_noisf${isf_out}_notblock${tblock_out}.sh

# to remove if no CV!!!
done






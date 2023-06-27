#!/bin/bash

#PREPARE NEMO DATA FOR USE FOR MY SCRIPTS - new mask/float

#homepath=/Users/claraburgard/bettik_clara
homepath=/bettik/burgardc

# name of NEMO run
#nemo_run=EPM026
nemo_run=EPM031
#nemo_run=EPM034


################# DECLARE THE PATHS ##############################
path2=$homepath/DATA/NN_PARAM/raw/PIERRE_COUPLED/eORCA025.L121-"$nemo_run"
path3=$homepath/DATA/NN_PARAM/interim/NEMO_eORCA025.L121_"$nemo_run"_ANT_STEREO
path5=$homepath/DATA/NN_PARAM/raw/
####################################################################

#for yy in {2049..2058} #EPM026
for yy in {2049..2058} #EPM031
#for yy in {2119..2128} #EPM034

do
cdo ifthenc,2 -subc,1 -selvar,tmask $path3/mask_variables_of_interest_Ant_"$yy".nc $path3/lsmask_0-2_Ant_withmiss.nc  
cdo setmisstoc,0 $path3/lsmask_0-2_Ant_withmiss.nc $path3/lsmask_0-2_Ant.nc 

cdo ifthenc,1 -selvar,isf_draft $path3/mask_variables_of_interest_Ant_"$yy".nc $path3/isfmask_1_Ant_withmiss.nc 
cdo setmisstoc,0 $path3/isfmask_1_Ant_withmiss.nc $path3/isfmask_1_Ant.nc 

cdo add $path3/lsmask_0-2_Ant.nc $path3/isfmask_1_Ant.nc $path3/lsmask_0-1-2_andsome3s_Ant.nc
cdo mul -lec,2 $path3/lsmask_0-1-2_andsome3s_Ant.nc $path3/lsmask_0-1-2_andsome3s_Ant.nc $path3/lsmask_0-1-2_Ant_nonfloat.nc
cdo mulc,1.0 $path3/lsmask_0-1-2_Ant_nonfloat.nc $path3/lsmask_0-1-2_Ant_"$yy".nc 
done

#cdo fldmean -selvar,gdept_0 $path3/mask_variables_of_interest_Ant.nc $path3/mean_depth_coord_Ant.nc <= cannot be used if grid area weights are not there

### go to notebook 'custom_lsmask.ipynb'
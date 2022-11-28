#!/bin/bash

#########
#This script is to format the raw NEMO data for further use 
#It extracts the interesting variables and adds the right grid information needed to regrid it in the next step
########

homepath=/bettik/burgardc

#### name of NEMO run
# nemo_run=OPM006
# nemo_run=OPM016
# nemo_run=OPM018
# nemo_run=OPM021


################# DECLARE THE PATHS ##############################
path1=$homepath/DATA/NN_PARAM/raw/SMITH_DATA
path2=$homepath/DATA/NN_PARAM/interim/SMITH_bi646
#path3=$homepath/DATA/BASAL_MELT_PARAM/interim/NEMO_eORCA025.L121_"$nemo_run"_ANT_STEREO
#path4=$homepath/DATA/BASAL_MELT_PARAM/raw/BEDMACHINE_RTOPO
path5=$homepath/DATA/NN_PARAM/raw/
###################################################################

###### VARIABLES

echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien.nc $path2/variables_of_interest_allyy.nc

for var in {thetao,so}
do
echo $var
echo 'ncks > extract variable from gridT' $var
ncks -O -C -v $var $path1/bi646_1y_grid-T-y300tMX_ANNAVG.nc $path2/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/$var.nc $path2/variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/variables_of_interest_allyy.nc
\rm $path2/$var.nc
done

var=sowflisf
echo $var
echo 'ncks > extract variable from flxT' $var
ncks -O -C -v $var $path1/bi646_1y_isf-T-y300tMX_ANNAVG.nc $path2/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/$var.nc $path2/variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/variables_of_interest_allyy.nc
\rm $path2/$var.nc
done

###### MASK

echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien.nc $path2/mask_variables_of_interest_allyy.nc

# write out the ice-shelf draft and the bathymetry 
for var in {isf_draft,Bathymetry_isf} 
do
echo $var
echo 'ncks > extract variable'
ncks -O -C -v $var $path1/bi646c_bathymetry-isf-y300.nc  $path2/$var.nc
echo 'ncks > put variable in file'
ncks -A -C -v $var $path2/$var.nc $path2/mask_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat'
ncatted -a coordinates,$var,m,c,"lon lat" $path2/mask_variables_of_interest_allyy.nc


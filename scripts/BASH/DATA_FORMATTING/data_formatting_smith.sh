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
cp $path5/grid_eORCA025_CDO_Fabien.nc $path2/variables_of_interest_allyy.nc # I would need to check if I can cut the grid

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
done

#### PREPARE LAND SEA MASK

cdo sub Bathymetry_isf.nc isf_draft.nc diff_bathy_draft.nc # difference bathymetry - isf draft
cdo gtc,0 isf_draft.nc isfdraft_gtc0.nc # identify where there is ice => 1
cdo eqc,0 Bathymetry_isf.nc bathy_0.nc # identify where there is ground without ice => 1
cdo add bathy_0.nc isfdraft_gtc0.nc ice_1.nc # where there is ice and ground without ice => 1
cdo eqc,0 diff_bathy_draft.nc diff_0.nc # where there is ice and ground without ice => 1
cdo add ice_1.nc diff_0.nc lsmask_012.nc # land sea mask with 012
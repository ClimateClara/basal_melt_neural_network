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
# nemo_run=bf663
nemo_run=bi646

################# DECLARE THE PATHS ##############################
path1=$homepath/DATA/NN_PARAM/raw/SMITH_DATA
path2=$homepath/DATA/NN_PARAM/interim/SMITH_"$nemo_run"
#path3=$homepath/DATA/BASAL_MELT_PARAM/interim/NEMO_eORCA025.L121_"$nemo_run"_ANT_STEREO
#path4=$homepath/DATA/BASAL_MELT_PARAM/raw/BEDMACHINE_RTOPO
path5=$homepath/DATA/NN_PARAM/raw
###################################################################



###### VARIABLES
ncks -d y,0,500 $path5/grid_eORCA025_CDO_Fabien.nc -o $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc

echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc $path2/3D_variables_of_interest_allyy.nc 

for var in {thetao,so}
do
echo $var
echo 'ncks > extract variable from gridT' $var
#ncks -O -C -v $var $path1/bi646_1y_grid-T-y300tMX_ANNAVG.nc $path2/$var.nc
ncks -O -C -v $var $path1/"$nemo_run"_1y_grid-T_AVG.nc $path2/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/$var.nc $path2/3D_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/3D_variables_of_interest_allyy.nc
\rm $path2/$var.nc
done

echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc $path2/2D_variables_of_interest_allyy.nc 

var=sowflisf
echo $var
echo 'ncks > extract variable from gridT' $var
ncks -O -C -v $var $path1/"$nemo_run"_1y_isf-T_AVG.nc $path2/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/$var.nc $path2/2D_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/2D_variables_of_interest_allyy.nc
\rm $path2/$var.nc

var=tos
echo $var
echo 'ncks > extract variable from gridT' $var
ncks -O -C -v $var $path1/"$nemo_run"_1y_grid-T_AVG.nc $path2/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/$var.nc $path2/2D_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/2D_variables_of_interest_allyy.nc
\rm $path2/$var.nc


echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc $path2/3D_depth_coord.nc 

var=gdept_0
echo $var
echo 'ncks > extract variable from mesh_mask' $var
ncks -O -C -v $var $path1/nemo_"$nemo_run"c_18801201_mesh_mask.nc $path2/$var.nc # add c if bi646, o if bf663
echo 'cut region of interest'
ncks -d y,0,500 $path2/$var.nc $path2/"$var"_cut.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/"$var"_cut.nc $path2/3D_depth_coord.nc 
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/3D_depth_coord.nc 
\rm $path2/$var.nc
\rm $path2/"$var"_cut.nc


###### MASK

echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc $path2/mask_variables_of_interest_allyy.nc

# write out the ice-shelf draft and the bathymetry 
for var in {isf_draft,Bathymetry_isf} 
do
echo $var
echo 'ncks > extract variable'
ncks -O -C -v $var $path1/bi646c_YYYY1201_bathymetry-isf.nc  $path2/$var.nc
echo 'ncks > put variable in file'
ncks -A -C -v $var $path2/$var.nc $path2/mask_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat'
ncatted -a coordinates,$var,m,c,"lon lat" $path2/mask_variables_of_interest_allyy.nc
done

cdo setgrid,$path2/2D_variables_of_interest_allyy.nc $path2/mask_variables_of_interest_allyy.nc $path2/mask_variables_of_interest_allyy_setgrid.nc
cdo sellonlatbox,0,360,-90,-50 $path2/mask_variables_of_interest_allyy_setgrid.nc $path2/mask_variables_of_interest_allyy_Ant.nc
cdo sellonlatbox,0,360,-90,-50 $path2/3D_variables_of_interest_allyy.nc $path2/3D_variables_of_interest_allyy_Ant.nc
cdo sellonlatbox,0,360,-90,-50 $path2/2D_variables_of_interest_allyy.nc $path2/2D_variables_of_interest_allyy_Ant.nc
cdo setgrid,$path2/2D_variables_of_interest_allyy.nc $path2/3D_depth_coord.nc $path2/3D_depth_coord_setgrid.nc
cdo sellonlatbox,0,360,-90,-50 $path2/3D_depth_coord_setgrid.nc $path2/3D_depth_coord_Ant.nc 


#### PREPARE LAND SEA MASK

cdo sub -selvar,Bathymetry_isf $path2/mask_variables_of_interest_allyy_Ant.nc -selvar,isf_draft $path2/mask_variables_of_interest_allyy_Ant.nc  $path2/diff_bathy_draft.nc # difference bathymetry - isf draft
cdo gtc,0 -selvar,isf_draft $path2/mask_variables_of_interest_allyy_Ant.nc $path2/isfdraft_gtc0.nc # identify where there is ice => 1
cdo eqc,0 -selvar,Bathymetry_isf $path2/mask_variables_of_interest_allyy_Ant.nc $path2/bathy_0.nc # identify where there is ground without ice => 1
cdo add $path2/bathy_0.nc $path2/isfdraft_gtc0.nc $path2/ice_1.nc # where there is ice and ground without ice => 1
cdo eqc,0 $path2/diff_bathy_draft.nc $path2/diff_0.nc # where there is ice and ground without ice => 1
cdo add $path2/ice_1.nc $path2/diff_0.nc $path2/lsmask_012.nc # land sea mask with 012
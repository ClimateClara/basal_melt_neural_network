#!/bin/bash

#########
#This script is to format the raw NEMO data for further use 
#It extracts the interesting variables and adds the right grid information needed to regrid it in the next step
########

homepath=/bettik/burgardc

#### name of NEMO run
#nemo_run=EPM026 # control
#nemo_run=EPM031 # perturbation
nemo_run=EPM034 # reversibility

################# DECLARE THE PATHS ##############################
path1=$homepath/DATA/NN_PARAM/raw/PIERRE_COUPLED
path2=$homepath/DATA/NN_PARAM/interim/PIERRE_"$nemo_run"
path3=$homepath/DATA/NN_PARAM/interim/NEMO_eORCA025.L121_"$nemo_run"_ANT_STEREO
#path4=$homepath/DATA/BASAL_MELT_PARAM/raw/BEDMACHINE_RTOPO
path5=$homepath/DATA/NN_PARAM/raw
###################################################################

############### WRITE OUT ONLY VARIABLES OF INTEREST #################
#for yy in {2049..2058} #EPM026
#for yy in {2049..2058} #EPM031
for yy in {2119..2128} #EPM034

do
echo $yy
cdo selvar,votemper,vosaline,sosst $path1/eORCA025.L121-"$nemo_run"/eORCA025.L121-"$nemo_run"_y"$yy".1y_gridT.nc $path2/eORCA025.L121-"$nemo_run"_y"$yy".1y_gridT_varofint.nc 
cdo selvar,sowflisf_cav $path1/eORCA025.L121-"$nemo_run"/eORCA025.L121-"$nemo_run"_y"$yy".1y_flxT.nc $path2/eORCA025.L121-"$nemo_run"_y"$yy".1y_flxT_varofint.nc 
cp $path1/eORCA025.L121-"$nemo_run"/eORCA025.L121-"$nemo_run"_domain_cfg_reduced_"$yy"*.nc $path2/eORCA025.L121-"$nemo_run"_mask_"$yy".nc
done


############### DATA MANIPULATION TO HAVE IT ON THE RIGHT GRID FOR CDO #################
# -O overwriting file
# -C -v control variables
# -a no alphabetical order of the variables

#for yy in {2049..2058} #EPM026
#for yy in {2049..2058} #EPM031
for yy in {2119..2128} #EPM034


do

echo $yy
echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien.nc $path3/variables_of_interest_"$yy".nc

for var in {votemper,vosaline,sosst}
do
echo $var
echo 'ncks > extract variable from gridT' $var
ncks -O -C -v $var $path2/eORCA025.L121-"$nemo_run"_y"$yy".1y_gridT_varofint.nc $path3/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path3/$var.nc $path3/variables_of_interest_"$yy".nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path3/variables_of_interest_"$yy".nc
\rm $path3/$var.nc
done

var=sowflisf_cav
echo $var
echo 'ncks > extract variable from flxT' $var
ncks -O -C -v $var $path2/eORCA025.L121-"$nemo_run"_y"$yy".1y_flxT_varofint.nc $path3/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path3/$var.nc $path3/variables_of_interest_"$yy".nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path3/variables_of_interest_"$yy".nc
\rm $path3/$var.nc


echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien.nc $path3/mask_variables_of_interest_"$yy".nc

for var in {isf_draft,bathy_metry,e3w_0} #,e3t_0
do
echo $var
echo 'ncks > extract variable'
ncks -O -C -v $var $path2/eORCA025.L121-"$nemo_run"_mask_"$yy".nc $path3/$var.nc
echo 'ncks > put variable in file'
ncks -A -C -v $var $path3/$var.nc $path3/mask_variables_of_interest_"$yy".nc
echo 'ncatted > put coords to lon lat'
ncatted -a coordinates,$var,m,c,"lon lat" $path3/mask_variables_of_interest_"$yy".nc
done

var=tmask #instead of tmaskutil to avoid the vertical line
echo $var 
echo 'define land sea mask'
cdo ifthenc,1 -vertsum -selvar,tmask $path2/eORCA025.L121-"$nemo_run"_mask_"$yy".nc $path3/"$var"_sum_withmiss.nc # setmisstoc,0 -
cdo setmisstoc,0 $path3/"$var"_sum_withmiss.nc $path3/"$var"_sum.nc
echo 'ncks > extract variable'
ncks -O -C -v $var $path3/"$var"_sum.nc $path3/$var.nc
echo 'ncks > put variable in file'
ncks -A -C -v $var $path3/$var.nc $path3/mask_variables_of_interest_"$yy".nc
echo 'ncatted > put coords to lon lat'
ncatted -a coordinates,$var,m,c,"lon lat" $path3/mask_variables_of_interest_"$yy".nc

done

#for yy in {2049..2058} #EPM026
#for yy in {2049..2058} #EPM031
for yy in {2119..2128} #EPM034

do

echo $yy
echo 'cut variable file'
cdo sellonlatbox,0,360,-90,-50 $path3/variables_of_interest_"$yy".nc $path3/variables_of_interest_"$yy"_Ant.nc
echo 'cut mask file'
#cdo setgrid,$path3/variables_of_interest_2049.nc $path3/mask_variables_of_interest_"$yy".nc $path3/mask_variables_of_interest_setgrid_"$yy".nc
cdo setgrid,$path3/variables_of_interest_2119.nc $path3/mask_variables_of_interest_"$yy".nc $path3/mask_variables_of_interest_setgrid_"$yy".nc
cdo sellonlatbox,0,360,-90,-50 $path3/mask_variables_of_interest_setgrid_"$yy".nc $path3/mask_variables_of_interest_Ant_"$yy".nc
done


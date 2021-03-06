# Sample data files

This directory contains sample data:

* `gchp13_sample_species.nc` is a sample dataset with a C48 gnomonic cubed-sphere grid
* `c48_gridspec/` contains has the C48 grid definition (in GRIDSPEC Mosaic file format; see [ref](https://earthsystemmodeling.org/docs/release/ESMF_8_1_1/ESMF_refdoc/node3.html#SECTION03028500000000000000)) 
* `regular_lat_lon_90x180.nc` is a grid definition file for a regular lat-lon grid (in CF Single Tile format; see [ref](https://earthsystemmodeling.org/docs/release/ESMF_8_1_1/ESMF_refdoc/node3.html#sec:fileformat:gridspec))
* `esmf_regrid_weights_c48_to_latlon90x180.nc` are regridding weights generated by [`ESMF_RegridWeightGen`](https://earthsystemmodeling.org/docs/release/ESMF_8_1_1/ESMF_refdoc/node3.html#SECTION03020000000000000000)

This sample data is suitable for demonstrating cubed-sphere to lat-lon regridding. Regridding between lat-lon,
gaussian, cubed-sphere, stretched cubed-sphere grids, etc. follows the same procedure.


The grid of our dataset is defined by the GRIDSPEC files in `c48_gridspec/`. These files were created with the 
following [gridspec](https://github.com/LiamBindle/gridspec) command: 
```console
$ gridspec-create gcs 48
Creating gnomonic cubed-sphere grid.
  Cubed-sphere size: C48

Writing mosaic and tile files
  + c48_gridspec.nc
  + c48.tile1.nc
  + c48.tile2.nc
  + c48.tile3.nc
  + c48.tile4.nc
  + c48.tile5.nc
  + c48.tile6.nc

Created 7 files.
$
```

In this demo, the desired output grid is a simple 2°x2° grid, which is defined by `regular_lat_lon_90x180.nc`. This
file was generated with the following [gridspec](https://github.com/LiamBindle/gridspec) command:
```console
$ gridspec-create latlon 90 180
Creating regular lat-lon grid.
  Latitude dimension:  90
  Longitude dimension: 180
  Pole-centered:       False
  Dateline-centered:   False

Writing mosaic and tile files.
  + regular_lat_lon_90x180.nc

Created 1 file.
$
```

To regrid the input dataset to the 2°x2° grid we need the regridding weights (a sparse matrix that defines the 
remapping). These weights are defined in `esmf_regrid_weights_c48_to_latlon90x180.nc` which was generated with 
the following [`ESMF_RegridWeightGen`](https://earthsystemmodeling.org/docs/release/ESMF_8_1_1/ESMF_refdoc/node3.html#SECTION03020000000000000000) command:
```console
$ ESMF_RegridWeightGen -s c48_gridspec/c48_gridspec.nc -d regular_lat_lon_90x180.nc -m conserve -w esmf_regrid_weights_c48_to_latlon90x180.nc --tilefile_path c48_gridspec/
 Starting weight generation with these inputs: 
   Source File: c48_gridspec/c48_gridspec.nc
   Destination File: regular_lat_lon_90x180.nc
   Weight File: esmf_regrid_weights_c48_to_latlon90x180.nc
   Source File is in GRIDSPEC MOSAIC format
   Use the center coordinates of the source grid to do the regrid
   Destination File is in CF Grid format
   Destination Grid is a global grid
   Destination Grid is a logically rectangular grid
   Use the center coordinates of the destination grid to do the regrid
   Regrid Method: conserve
   Pole option: NONE
   Line Type: greatcircle
   Norm Type: dstarea
   Extrap. Method: none
   Alternative tile file path: c48_gridspec/

 Completed weight generation successfully.

$
```

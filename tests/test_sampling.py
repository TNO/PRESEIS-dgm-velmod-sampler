from pathlib import Path
import numpy as np
import xarray as xr
import pyproj as prj
import sys

test_path = Path(__file__).parent
module_path = test_path.parent
sys.path.insert(0, str(module_path))

from dgm_velmod_sampler import sample_dgm_velmod

sample_path = test_path / "res/sample.h5"
velmod_path = test_path / "res/VELMOD31_UTM31.h5"
dgm_path = test_path / "res/DGM5_UTM31.h5"


def test_config_file_exists():
    assert (module_path / "config/config.json").exists() == True


def test_sampling(create=False):
    assert velmod_path.exists()
    assert dgm_path.exists()
    if not create:
        assert sample_path.exists()

    velmod = xr.load_dataset(velmod_path, decode_coords="all")
    dgm = xr.load_dataset(dgm_path, decode_coords="all")

    crs_UTM, crs_RD = (
        prj.CRS("EPSG:23031"),
        prj.CRS("EPSG:28992"),
    )
    RD_to_UTM = prj.Transformer.from_crs(crs_RD, crs_UTM)

    # Define grid in RD
    # X,Y samples
    xsmp_RD = np.linspace(100000.0, 110000.0, 41)
    ysmp_RD = np.linspace(450000.0, 460000.0, 41)
    zsmp = np.linspace(-5000.0, 0.0, 51)

    # Prepare xarray representation, without data, just coordinates
    grid = (
        xr.Dataset(coords={"x": xsmp_RD, "y": ysmp_RD, "z": zsmp})
        .rio.write_crs(crs_RD.to_epsg())
        .rio.write_coordinate_system()
    )

    # Determine UTM coordinates for all grid points
    x_UTM, y_UTM = xr.apply_ufunc(
        RD_to_UTM.transform,
        grid["x"],
        grid["y"],
        output_core_dims=[[], []],
        vectorize=True,  # prj Transformers do not broadcast
        keep_attrs=True,
    )

    # Sample models to cube, no need to pass CRS since it is represented in the x_UTM data structure
    dgm_velmod_cube = sample_dgm_velmod(x_UTM, y_UTM, grid["z"], dgm=dgm, velmod=velmod)

    if create:
        dgm_velmod_cube.to_netcdf(sample_path, mode="w")

    sample = xr.load_dataset(sample_path, decode_coords="all")

    assert dgm_velmod_cube == sample

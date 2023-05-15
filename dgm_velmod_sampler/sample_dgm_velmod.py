import numpy as np
import rioxarray
import xarray as xr


def sample_dgm_velmod(
    x, y, z, dgm: xr.Dataset, velmod: xr.Dataset, crs=None
) -> xr.Dataset:
    """
    Create a collection of (interpolated) samples of the DGM/VELMOD optionally including depth samples.

    Parameters
    ----------
    x, y : array_like
        Spatial coordinates of points where interpolation of DGM and VELMOD is requested.
        The (`x`,`y`) coordinates are assumed to be in the same coordinate reference system as specified in
        the `velmod` and `dgm` models supplied in the arguments. The `x` and `y` values van be arranged in
        an arbitrary number of dimensions, most conveniently in an xarray structure. The dimensions and
        coordinates of the `x` and `y` structures can have an arbitrary logical meaning structure, such
        as a simple list of locations, or a grid structure. The grid structure may have its own
        coordinate system, such as (inline, x-line) or spatial coordinates in a certain coordinate
        reference system (CRS). Oftentimes, the CRS may be different from the CRS of the DGM/Velmod model.
        Any CRS specified in a `x.crs` member (as by rioxarray) will be honoured, unless overwritten by
        the optional argument. Any `y.crs` will be ignored.

    z : array_like or None
        Array of vertical locations where instantaneous velocities are requested. The vertical reference is
        NAP (sea level). Negative is down. If `z` equals None, the DGM and VELMOD maps will be interpolated
        without an explicit extrapolation to depth.

    dgm: xarray.Dataset
        Represents the DGM-5 model as generated by the convert.py script

    velmod: xarray.Dataset
        Represents the VELMOD-3.1 model as generated by the convert.py script

    crs: Any, Optional
        A coordinate reference system (CRS) specifier accepted by rioxarray. Represent the CRS for the
        coordinates of the `x` and `y` grid dimensions; not the CRS of the `x` and `y` values, which
        should match the `dgm` and `velmod` CRS's.

    Returns
    -------
    xarray.Dataset
        Dataset with the interpolated DGM and VELMOD data and (optional) vertical depth samples

    """

    # assert the coordinate reference system of dgm and velmod is equal
    assert (
        dgm.rio.crs == velmod.rio.crs
    ), "coordinate reference systems of dgm and velmod do not match"

    # determine which units are common to both datasets
    sel_units = np.intersect1d(dgm["unit"], velmod["unit"])

    # interpolate VELMOD
    velmod_itp = (
        velmod.sel({"unit": sel_units}).interp({"x": x, "y": y}).sortby("ordering")
    )

    # interpolate DGM
    dgm_itp = dgm.sel({"unit": sel_units}).interp({"x": x, "y": y}).sortby("ordering")

    # generate combined velocity grid
    # for this only simply kriging is available ("sk")
    V0 = velmod_itp["V0_filled"].sel(
        {"summary_statistic": "mean", "kriging_type": "sk"}, drop=True
    )
    SD = velmod_itp["V0_filled"].sel(
        {"summary_statistic": "sd", "kriging_type": "sk"}, drop=True
    )

    dgm_velmod_samples = xr.Dataset(
        {
            "V0": V0,
            "V0_sd": SD,
            "k": velmod_itp["k"],
            "depth": dgm_itp["tvd"],
            "ordering": dgm_itp["ordering"],
        }
    )

    # create full depth grids for all units
    # note that this is not super efficient since only a limited section will
    # be used in the final result
    # Vinst = V0 + k * z
    if z is not None:
        # create depth profiles for each unit
        unit_vinst = V0 - velmod_itp["k"] * z

        # determine which unit is present at which depth
        unit = (dgm_itp["tvd"] < z).idxmax("unit")
        dgm_velmod_samples["unit_samples"] = unit

        # create corresponding mask
        unit_mask = unit_vinst["unit"] == unit
        dgm_velmod_samples["unit_mask"] = unit_mask

        # and combine both
        dgm_velmod_samples["Vinst"] = unit_vinst.where(unit_mask, 0.0).sum("unit")

    # check if the CRS is stored in the inputs already
    if not crs:
        try:
            crs = x.rio.crs
        except:
            pass

    if crs:
        dgm_velmod_samples.rio.write_crs(crs, inplace=True)
        dgm_velmod_samples.rio.write_coordinate_system(inplace=True)

    return dgm_velmod_samples

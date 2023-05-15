""" 

This script converts the zmap files of DGM5 and VELMOD3.1 into xarray format.
It uses the UTM31 zmap files as basis. Conversion to other coordinate reference
systems is easily accomplished after the conversion.
The zmap file locations are determined by the settings in `config.json`. All files will be 
accesibly after running the `configure.py` script."

"""

import math
import json5
from pathlib import Path
import xarray as xr
import pandas as pd
from tqdm import tqdm
import zmapio  # https://pypi.org/project/zmapio/
import rioxarray  # provide rio accessor to xarray for georeferencing

# we use the UTM version of DGM and VELMOD as it appears to be the original and most consistent
crs_UTM31 = "epsg:23031"  # https://epsg.io/23031


def convert(verbose=False):
    module_path = Path(__file__).parent.parent
    base_path = module_path

    config_path = module_path / "config/config.json"
    with open(config_path, "r") as f:
        config = json5.load(f)

    out_path = Path(config["output_directory"])
    download_path = Path(config["download_directory"])
    velmod31_k_file = Path(config["velmod31_k_file"])

    if not download_path.is_absolute():
        download_path = base_path / download_path
    if not out_path.is_absolute():
        out_path = base_path / out_path
    if not velmod31_k_file.is_absolute():
        velmod31_k_file = base_path / velmod31_k_file

    out_path.mkdir(parents=True, exist_ok=True)
    out_velmod = out_path / "VELMOD31_UTM31.h5"
    out_dgm = out_path / "DGM5_UTM31.h5"

    velmod_zmap_list = list(download_path.glob("velmod31/**/*.dat"))
    dgm_zmap_list = list(download_path.glob("dgmdeep5/**/*tvd*merge_*UTM31.zmap"))

    # hard code ordering of units
    unit_canonical_order = [
        "N",
        "NU",
        "NLNM",
        "NM",
        "NL",
        "CK",
        "KN",
        "KNG",
        "KNGL",
        "KNN",
        "S",
        "SL",
        "SG",
        "SK",
        "ATPO",
        "AT",
        "TR",
        "RN",
        "RB",
        "ZE",
        "RO",
        "DCC",
        "DC",
        "CL",
    ]

    # VELMOD3.1
    print("converting VELMOD3.1 to h5")

    # read csv with k values as extracted from documentation
    velmoddata = xr.Dataset(pd.read_csv(velmod31_k_file, delimiter=";", index_col=1))

    # select and read all data files
    velmodds = [velmod_zmap_to_xarray(zm, crs_UTM31) for zm in tqdm(velmod_zmap_list)]

    # concatenate individual files
    velmodxrc = (
        xr.concat(velmodds, "u_v_k_s")
        .set_index(
            {"u_v_k_s": ["unit", "variable", "kriging_type", "summary_statistic"]}
        )
        .unstack()
        .dropna("x", "all")
        .dropna("y", "all")
    )

    # partition "variable" dimension into separate datasets
    velmodxrds = xr.merge([velmodxrc.to_dataset("variable"), velmoddata])

    # add canonical ordering (top to bottom) and sort accordingly
    ordering = xr.zeros_like(velmodxrds["unit"], dtype=int).rename("ordering")
    for u in ordering.coords["unit"]:
        ordering.loc[{"unit": u}] = unit_canonical_order.index(u)
    velmod_UTM = xr.merge((velmodxrds, ordering)).sortby("ordering")

    # integrate ZE in the Vinst=V0+k*z template by setting V0=Vint and utlizing k=0 for ZE set above
    # this is for sampling convenience
    velmod_UTM["V0"].loc[{"unit": "ZE"}] = velmod_UTM["Vint"].loc[{"unit": "ZE"}]

    # Also create grids with the missing parts filled in by the unit mean.
    # This turns out to be necessary when combining DGM and VELMOD due to different sampling and also interpolation issues.
    # Of course the filling could be more advanced. More appropriate geostatistical methods should be used.
    # However, this information should come from the VELMOD development team
    velmod_UTM["V0_filled"] = velmod_UTM["V0"].fillna(velmod_UTM["V0"].mean(["x", "y"]))

    # write to disk
    velmod_UTM.to_netcdf(out_velmod, mode="w")
    if verbose:
        print(f"wrote: {out_velmod.absolute()}")

    # DGM5
    print("converting DGM5 to h5")

    dgmds = [dgm_zmap_to_xarray(zm, crs_UTM31) for zm in tqdm(dgm_zmap_list)]

    # add bottom to DC at -infinity for convenience
    last = xr.full_like(dgmds[0], -math.inf)
    last["unit"] = "DC"
    dgmds.append(last)

    # concatenate individual files
    dgmxrc = (
        xr.concat(dgmds, "unit_var")
        .set_index({"unit_var": ["unit", "var"]})
        .unstack()
        .dropna("x", "all")
        .dropna("y", "all")
    )

    # partition "var" dimension into separate datasets
    dgmxrds = dgmxrc.to_dataset("var")

    # add canonical ordering (top to bottom) and sort accordingly
    dgm_ordering = xr.zeros_like(dgmxrds["unit"], dtype=int).rename("ordering")
    for u in dgm_ordering.coords["unit"]:
        dgm_ordering.loc[{"unit": u}] = unit_canonical_order.index(u)
    dgm_UTM = xr.merge([dgmxrds, dgm_ordering]).sortby("ordering")

    dgm_UTM.to_netcdf(out_dgm, mode="w")
    if verbose:
        print(f"wrote: {out_dgm.absolute()}")


def dgm_zmap_to_xarray(zmap_file, crs):
    name = zmap_file.stem
    name_list = name.split("_")
    unit = name_list[0]
    var = name_list[1]

    zm = zmapio.ZMAPGrid(zmap_file.as_posix())

    zmds = xr.Dataset.from_dataframe(zm.to_pandas())
    zmda = zmds.set_index({"index": ["X", "Y"]}).unstack()["Z"].T
    zmda = (
        zmda.expand_dims("unit_var")
        .assign_coords(
            {
                "unit": ("unit_var", [unit]),
                "var": ("unit_var", [var.strip("_")]),
            }
        )
        .rename({"X": "x", "Y": "y"})
        .rio.write_crs(crs)
        .rio.write_coordinate_system()
        .assign_attrs(
            {
                "model": "DGM5",
            }
        )
    )

    return zmda


def velmod_zmap_to_xarray(zmap_file, crs):
    name = zmap_file.stem.replace("NLM", "NLNM")  # tidying up
    # add suffix for mean, to distinguish from sd
    # the suffix is subsequently dropped for sd by selecting [0:5]
    name = name + "_mean"
    name_list = name.split("_")
    unit = name_list[0]
    var = name_list[2]
    kriging_type = name_list[3]
    summary_statistic = name_list[4]

    zm = zmapio.ZMAPGrid(zmap_file.as_posix())

    zmds = xr.Dataset.from_dataframe(zm.to_pandas())
    zmda = zmds.set_index({"index": ["X", "Y"]}).unstack()["Z"].T
    zmda = (
        zmda.expand_dims("u_v_k_s")
        .assign_coords(
            {
                "unit": ("u_v_k_s", [unit]),
                "variable": ("u_v_k_s", [var]),
                "kriging_type": ("u_v_k_s", [kriging_type]),
                "summary_statistic": ("u_v_k_s", [summary_statistic]),
            }
        )
        .rename({"X": "x", "Y": "y"})
        .rio.write_crs(crs)
        .rio.write_coordinate_system()
        .assign_attrs(
            {
                "model": "VELMOD3.1",
            }
        )
    )

    return zmda


if __name__ == "__main__":
    convert(verbose=True)

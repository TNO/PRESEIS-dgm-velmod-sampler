from pathlib import Path
import json5
import sys
import xarray as xr

test_path = Path(__file__).parent
module_path = test_path.parent
sys.path.insert(0, str(module_path))

import scripts.convert
import scripts.configure

velmod_ref_path = test_path / "res/VELMOD31_UTM31.h5"
dgm_ref_path = test_path / "res/DGM5_UTM31.h5"


def test_config_file_exists():
    assert (module_path / "config/config.json").exists() == True


def test_conversion(create=False):
    with open(module_path / "config/config.json", "r") as f:
        config = json5.load(f)

    scripts.configure.configure()
    scripts.convert.convert()

    velmod_path = Path(config["output_directory"]) / "VELMOD31_UTM31.h5"
    dgm_path = Path(config["output_directory"]) / "DGM5_UTM31.h5"
    if not velmod_path.is_absolute():
        velmod_path = module_path / velmod_path
    if not dgm_path.is_absolute():
        dgm_path = module_path / dgm_path

    assert velmod_path.exists()
    assert dgm_path.exists()

    velmod = xr.load_dataset(velmod_path, decode_coords="all")
    velmod_ref = xr.load_dataset(velmod_ref_path, decode_coords="all")
    assert velmod == velmod_ref

    dgm = xr.load_dataset(dgm_path, decode_coords="all")
    dgm_ref = xr.load_dataset(dgm_ref_path, decode_coords="all")
    assert dgm == dgm_ref

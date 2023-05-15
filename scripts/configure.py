""" 

This script download and unzips the zmap files of DGM5 and VELMOD3.1, such that
the `convert.py`script can pick them up and convert them into xarray format.
The file URL's are supplied in the `config.json` file included in the distribution.

"""

import json5
import wget
import zipfile
from pathlib import Path


def configure(verbose=False):
    module_path = Path(__file__).parent.parent
    config_path = module_path / "config/config.json"
    with open(config_path, "r") as f:
        config = json5.load(f)

    downloads = Path(config["download_directory"])
    if not downloads.is_absolute():
        downloads = module_path / downloads

    if verbose:
        print("download model files")
    filelist = []
    for ds, files in config["downloads"].items():
        folder = downloads / ds
        folder.mkdir(parents=True, exist_ok=True)
        assert folder.is_dir()
        for url in files:
            filename = folder / wget.detect_filename(url)
            if filename.exists():
                if verbose:
                    print(f"    file {filename} exists .. skipped")
            else:
                if verbose:
                    print(f"    downloading {filename}")
                wget.download(url, str(folder))
            filelist.append(filename)

    if verbose:
        print("extract ZMAP files")
    for zipf in filelist:
        folder = zipf.parent
        with zipfile.ZipFile(zipf) as z:
            files = z.namelist()
            for f in files:
                if f.endswith(("on_offshore_merge_DGM50_ED50_UTM31.zmap", ".dat")):
                    target = folder / f
                    if target.exists():
                        if verbose:
                            print(f"    file {target} exists .. skipped")
                    else:
                        if verbose:
                            print(f"    extracting {target}")
                        z.extract(f, path=folder)


if __name__ == "__main__":
    configure(verbose=True)

# GDAL Installation

The easiest path on all platforms is conda:

```bash
conda install -c conda-forge gdal
```

To verify the install worked:

```bash
python -c "from osgeo import gdal; print(gdal.__version__)"
```

## Linux (apt)

```bash
sudo apt-get install gdal-bin libgdal-dev
pip install gdal==$(gdal-config --version)
```

## macOS (Homebrew)

```bash
brew install gdal
pip install gdal==$(gdal-config --version)
```

## Using a Jupyter kernel with this environment

```bash
conda info --envs          # confirm environment name
conda activate crop-cls
python -m ipykernel install --user --name crop-cls --display-name "Python (crop-cls)"
```

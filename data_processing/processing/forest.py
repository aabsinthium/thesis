import os
import numpy as np
import xarray as xr

from utils import CustomCRS
from rasterio.enums import Resampling


class Forest:
    def __init__(self, year):
        self.year = year
        self.crs = CustomCRS().get_crs()

    @staticmethod
    def get_forest_flags(da):
        arr = da.to_numpy()
        classes = da.flag_values[6:25]

        arr[np.isin(arr, classes)] = 1
        arr[arr != 1] = 0

        return arr

    def get_forest(self):
        year = self.year
        new_crs = self.crs

        root = '/Users/artembadmaev/IT/thesis'
        match_da = xr.open_dataset(f'{root}/data/temp/fwi/{year}.nc').FWI.isel(time=0)
        match_da = match_da.rio.write_crs(new_crs)

        if year == 2021:
            da = xr.open_dataset(f'{root}/data/temp/forest/land/2020.nc').lccs_class.sel(
                time='2020-01-01', lat=slice(83, 40))
            os.remove(f'{root}/data/temp/forest/land/2020.nc')
        else:
            da = xr.open_dataset(f'{root}/data/temp/forest/land/{year}.nc').lccs_class.sel(
                time=f'{year}-01-01', lat=slice(83, 40))
            os.remove(f'{root}/data/temp/forest/land/{year}.nc')

        da = da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
        da = da.rio.write_crs(4326)
        da = da.rio.reproject(dst_crs=new_crs)
        da = da.sel(x=slice(0, 174))

        da.values = self.get_forest_flags(da)

        da = da.astype('float')
        da = da.rename('forest_cover')

        rda = da.rio.reproject_match(match_da, resampling=Resampling.average)
        rda = rda.where(rda != 255, 0)

        rda.to_netcdf(f'{root}/data/temp/forest/{year}.nc')

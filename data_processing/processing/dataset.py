import os
import xarray as xr
import geopandas as gpd

from utils import CustomCRS


class Dataset:
    def __init__(self, year):
        self.year = year
        self.root = '/Users/artembadmaev/IT/thesis'

        self.crs = CustomCRS().get_crs()
        self.geo = gpd.read_file(f'{self.root}/data_processing/regions.json')
        self.geo = self.geo[self.geo.ID_1.isin(
            [4, 10, 13, 14, 21, 27, 31, 33, 40, 45, 51, 65, 75, 77, 87])]
        self.geo = self.geo.set_crs(4326)
        self.geo = self.geo.to_crs(self.crs)

    def load_ds(self, var):
        root = self.root
        year = self.year

        ds = xr.open_dataset(f'{root}/data/temp/{var}/{year}.nc')
        os.remove(f'{root}/data/temp/{var}/{year}.nc')

        return ds

    def clip(self, ds):
        geo = self.geo
        crs = self.crs

        ds = ds.rio.write_crs(crs)
        ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
        ds = ds.rio.clip(geo.geometry.values)

        ds = ds.drop('spatial_ref')

        for var in list(ds.data_vars):
            if 'grid_mapping' in ds[var].attrs.keys():
                del ds[var].attrs['grid_mapping']
        return ds

    def combine(self):
        root = self.root
        year = self.year

        fwi = self.load_ds('fwi').FWI
        forest = self.load_ds('forest').forest_cover.drop('spatial_ref')
        fires = self.load_ds('fires').counts

        ds = xr.merge([fwi, forest, fires])
        ds = self.clip(ds)

        ds.to_netcdf(f'{root}/data/{year}.nc')

        return ds

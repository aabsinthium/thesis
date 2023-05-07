import os
import numpy as np
import xarray as xr

from utils import CustomCRS
from rioxarray.merge import merge_datasets


class DataLoader:
    def __init__(self, year):
        self.year = year
        self.crs = CustomCRS().get_crs()
        self.root = '/Users/artembadmaev/IT/thesis'
        self.hours = ['06', '09', '12', '15', '18']

    def get_weather_ds(self):
        root = self.root
        year = self.year
        components = ['wind', 'humidity', 'temperature', 'precipitation']

        data = []
        new_crs = self.crs

        for component in components:
            weather = self.get_component(component)
            weather = weather.rio.write_crs(new_crs)
            weather = weather.rio.set_spatial_dims(x_dim='x', y_dim='y')

            data.append(weather)

        ds = xr.merge(data)
        ds['Month'] = (['time'], ds.time.dt.month.values)

        for var in list(ds.data_vars):
            if 'grid_mapping' in ds[var].attrs.keys():
                del ds[var].attrs['grid_mapping']

        ds.to_netcdf(f'{root}/data/temp/fwi/weather/{year}.nc')

        return ds

    def get_component(self, component):
        year = self.year
        root = self.root
        new_crs = self.crs

        path_east = f'{root}/data/temp/fwi/weather/weather_components/{component}/east/{year}'
        path_west = f'{root}/data/temp/fwi/weather/weather_components/{component}/west/{year}'

        if component == 'humidity':
            datasets = []
            hours = self.hours

            for hour in hours:
                datasets.append(
                    self.merge_weather(path_east, path_west, new_crs, hour=hour))

            ds = xr.merge(datasets)
            variables = list(ds.data_vars)

            arr = np.array([ds[variable].to_numpy() for variable in variables])
            arr = arr.min(axis=0)

            ds['Relative_Humidity_min'] = (['time', 'y', 'x'], arr)
            ds = ds.drop_vars(variables)
        else:
            ds = self.merge_weather(path_east, path_west, new_crs)

            if component == 'temperature':
                ds.Temperature_Air_2m_Mean_24h.values = ds.Temperature_Air_2m_Mean_24h.values - 273.15
            if component == 'wind':
                ds.Wind_Speed_10m_Mean.values = ds.Wind_Speed_10m_Mean.values * 3.6

        return ds

    @staticmethod
    def merge_weather(path_east, path_west, crs, hour=''):

        files_east = [f'{path_east}/{hour}/{file_name}'
                      for file_name in np.sort(os.listdir(f'{path_east}/{hour}'))
                      if file_name.endswith('.nc')]

        files_west = [f'{path_west}/{hour}/{file_name}'
                      for file_name in np.sort(os.listdir(f'{path_west}/{hour}'))
                      if file_name.endswith('.nc')]

        data = []
        files = list(zip(files_east, files_west))

        for file in files:
            east = xr.open_dataset(file[0])
            east = east.rio.write_crs(4326)
            east = east.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
            east = east.rio.reproject(crs)

            west = xr.open_dataset(file[1])
            west = west.rio.write_crs(4326)
            west = west.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
            west = west.rio.reproject(crs)

            ew = merge_datasets([east, west])
            data.append(ew)

            os.remove(file[0])
            os.remove(file[1])

        ds = xr.concat(data, dim='time')

        return ds

import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from utils import CustomCRS
from numpy.lib.stride_tricks import sliding_window_view


class Fires:
    def __init__(self, year):
        self.year = year
        self.crs = CustomCRS().get_crs()

    @staticmethod
    def get_edges(grid):
        grid = np.sort(grid.to_numpy())

        diff = np.diff(grid)
        diff = np.insert(diff, [0, -1], [-diff[0], diff[-1]]) / 2

        edges = np.insert(grid, 0, grid[0])
        edges += diff
        edges = np.hstack([sliding_window_view(edges, 2), grid.reshape(-1, 1)])

        return edges

    def get_counts(self, data, grid):
        lon_edges = self.get_edges(grid.x)
        lat_edges = self.get_edges(grid.y)

        gdf = data.copy()

        counts = xr.Dataset(
            data_vars={'counts': (['time', 'y', 'x'], np.zeros(grid.shape))},
            coords={'time': grid.time.values,
                    'y': lat_edges[:, 2],
                    'x': lon_edges[:, 2]})

        for lon_left, lon_right, lon_mid in lon_edges:
            if len(gdf) == 0:
                break

            slc = gdf.cx[lon_left:lon_right, :].copy()
            gdf.drop(slc.index, inplace=True)

            for lat_left, lat_right, lat_mid in lat_edges:
                if len(slc) == 0:
                    break

                slc_drop = slc.cx[:, lat_left:lat_right]

                if slc_drop.size == 0:
                    continue

                slc.drop(slc_drop.index, inplace=True)
                slc_drop = slc_drop.acq_date

                for date, count in slc_drop.value_counts().items():
                    counts.counts.loc[date, lat_mid, lon_mid] += count

        return counts

    def get_counts_ds(self):
        year = self.year
        new_crs = self.crs
        root = '/Users/artembadmaev/IT/thesis'

        gdf = pd.read_json(f'{root}/data/temp/fires/viirs/{year}.json')
        os.remove(f'{root}/data/temp/fires/viirs/{year}.json')

        gdf = gdf[['longitude', 'latitude', 'acq_date']]
        gdf.longitude = gdf.longitude - 18
        gdf.acq_date = pd.to_datetime(gdf.acq_date)
        gdf = gpd.GeoDataFrame(data=gdf,
                               geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude))

        da = xr.open_dataset(f'{root}/data/temp/fwi/{year}.nc').FWI

        counts = self.get_counts(gdf, da)

        counts = counts.rio.write_crs(new_crs)
        counts = counts.rio.set_spatial_dims(x_dim='x', y_dim='y')
        counts.to_netcdf(f'{root}/data/temp/fires/{year}.nc')

        return counts

import os
import pickle
import cdsapi
import zipfile


class APILoader:
    def __init__(self, year):
        self.year = year
        self.root = '/Users/artembadmaev/IT/thesis'
        self.hours = ['06', '09', '12', '15', '18']
        self.components = ['wind', 'humidity', 'temperature', 'precipitation']

    def get_land(self):
        year = self.year
        root = self.root

        path = f'{root}/data/temp/forest/land/{year}'
        dataset = 'satellite-land-cover'

        if year < 2016:
            request = f'{root}/data_processing/processing/api_requests/land_pre2016.pkl'
        else:
            request = f'{root}/data_processing/processing/api_requests/land_post2016.pkl'

        self.get_data(dataset, request, path)

        file = [file for file in os.listdir(path) if file.endswith('.nc')][0]
        os.replace(f'{path}/{file}', f'{path}.nc')

    def get_raw_weather(self):
        root = self.root
        dataset = 'sis-agrometeorological-indicators'

        year = self.year
        hours = self.hours
        components = self.components

        for comp in components:
            if comp == 'humidity':
                for hour in hours:
                    data_west = f'{root}/data/temp/fwi/weather/weather_components/{comp}/west/{year}/{hour}'
                    data_east = f'{root}/data/temp/fwi/weather/weather_components/{comp}/east/{year}/{hour}'

                    req_west = f'{root}/data_processing/processing/api_requests/{comp}_west_{hour}.pkl'
                    req_east = f'{root}/data_processing/processing/api_requests/{comp}_east_{hour}.pkl'

                    self.get_data(dataset, req_west, data_west)
                    self.get_data(dataset, req_east, data_east)
            else:
                data_west = f'{root}/data/temp/fwi/weather/weather_components/{comp}/west/{year}'
                data_east = f'{root}/data/temp/fwi/weather/weather_components/{comp}/east/{year}'

                req_west = f'{root}/data_processing/processing/api_requests/{comp}_west.pkl'
                req_east = f'{root}/data_processing/processing/api_requests/{comp}_east.pkl'

                self.get_data(dataset, req_west, data_west)
                self.get_data(dataset, req_east, data_east)

    def get_data(self, dataset, req_path, data_path):
        with open(req_path, 'rb') as f:
            request = pickle.load(f)

        request['year'] = self.year
        c = cdsapi.Client()

        os.makedirs(data_path, exist_ok=True)

        c.retrieve(
            dataset,
            request,
            f'{data_path}/data.zip')

        with zipfile.ZipFile(f'{data_path}/data.zip', 'r') as zf:
            zf.extractall(data_path)

        os.remove(f'{data_path}/data.zip')

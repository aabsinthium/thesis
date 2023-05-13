from argparse import ArgumentParser

from processing.fwi import FWI
from processing.cds import APILoader
from processing.fires import Fires
from processing.forest import Forest
from processing.weather import DataLoader
from processing.dataset import Dataset

parser = ArgumentParser()
parser.add_argument('-y', '--year', type=int, required=True)
parser.add_argument('-d', '--download', action='store_true')
parser.add_argument('--fwi', action='store_true')
parser.add_argument('--forest', action='store_true')
parser.add_argument('--fires', action='store_true')

args = parser.parse_args()

year = args.year

download = args.download
fwi_flag = args.fwi
forest_flag = args.forest
fires_flag = args.fires

if download:
    al = APILoader(year)
    al.get_raw_weather()
    al.get_land()

if fwi_flag:
    dl = DataLoader(year)
    dl.get_weather_ds()

    fwi = FWI(year)
    fwi.get_fwi()

if forest_flag:
    forest = Forest(year)
    forest.get_forest()

if fires_flag:
    fires = Fires(year)
    fires.get_counts_ds()

dataset = Dataset(year=year, region='rus')
dataset.combine()

from json import loads, dumps
from pyproj import CRS
from pyproj.crs.datum import CustomDatum, CustomPrimeMeridian


class CustomCRS:
    def __init__(self):
        crs = CRS.from_epsg(4326).to_json()
        crs_json = loads(crs)

        datum = CustomDatum(prime_meridian=CustomPrimeMeridian(18)).to_json()
        datum_json = loads(datum)

        custom_crs_json = crs_json
        custom_crs_json['datum'] = datum_json
        custom_crs_json = dumps(custom_crs_json)
        custom_crs = CRS.from_json(custom_crs_json)

        self.custom_crs = custom_crs

    def get_crs(self):
        return self.custom_crs

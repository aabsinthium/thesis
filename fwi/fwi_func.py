import numpy as np
import xarray as xr
from time import time
from datetime import datetime


def get_fwi_ds():
    return


def get_codes_ds(fpath, write_to=None):
    codes = ['ffmc', 'dmc', 'dc']
    ds = xr.open_dataset(fpath)

    for code in codes:
        print('Started processing {} at: {}'.format(code.upper(), datetime.now().time()))

        start = time()
        code_arr = get_code_arr(ds, code)
        end = time()

        print('{} ready. {:.2f} min elapsed'.format(code.upper(), (end - start) / 60))

        ds[code.upper()] = (['time', 'y', 'x'], code_arr)

    if write_to is not None:
        ds.to_netcdf(write_to)

    return ds


def get_code_arr(ds, code_name, initial_value=None):
    code_arr = []
    initials = {'ffmc': 85, 'dmc': 6, 'dc': 15}
    variables = {'ffmc': ['Temperature_Air_2m_Mean_24h', 'Relative_Humidity_min', 'Wind_Speed_10m_Mean', 'Precipitation_Flux'],
                 'dmc': ['Temperature_Air_2m_Mean_24h', 'Relative_Humidity_min', 'Precipitation_Flux', 'Month'],
                 'dc': ['Temperature_Air_2m_Mean_24h', 'Precipitation_Flux', 'Month']}

    x = len(ds.x)
    y = len(ds.y)
    time = len(ds.time)
    columns = len(variables[code_name])

    if initial_value is not None:
        initials[code_name] = initial_value

    da = ds[variables[code_name]].to_array().to_numpy()
    da = da.reshape(columns, time, -1)
    da = np.swapaxes(da, 0, 2)

    for arr in da:
        if np.all(np.isnan(arr)):
            code_arr.append(np.full([time], np.nan))
        else:
            code_arr.append(
                get_code_vec(arr, initials[code_name], code_name))

    code_arr = np.array(code_arr).reshape((y, x, -1))
    code_arr = np.moveaxis(code_arr, 2, 0)

    return code_arr


def get_code_vec(weather_array, initial_value, code_name):
    code_vec = [initial_value]

    for day in range(len(weather_array)):
        if code_name == 'ffmc':
            code_value = compute_ffmc(*weather_array[day], code_vec[day])
        elif code_name == 'dmc':
            code_value = compute_dmc(*weather_array[day], code_vec[day])
        elif code_name == 'dc':
            code_value = compute_dc(*weather_array[day], code_vec[day])
        else:
            raise NameError('Incorrect code name!')
        code_vec.append(code_value)

    return code_vec[1:]


def compute_ffmc(t, h, w, p, ffmc0):
    """Compute FFMC (Fine Fuel Moisture Code) given yesterday's FFMC value 'ffmc0',
     temperature, relative humidity, wind, and precipitation"""

    mo = (147.2 * (101 - ffmc0)) / (59.5 + ffmc0)

    if p > .5:
        rf = p - .5

        if mo > 150:
            mo = (mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))) \
                 + (.0015 * np.power(mo - 150, 2)) * np.sqrt(rf)
        else:
            mo = mo \
                 + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))

        if mo > 250:
            mo = 250

    ed = .942 * np.power(h, .679) \
        + (11 * np.exp((h - 100) / 10)) \
        + 0.18 * (21.1 - t) * (1 - 1 / np.exp(.1150 * h))

    if mo > ed:
        kl = .424 * (1 - np.power(h / 100, 1.7)) \
             + (.0694 * np.sqrt(w)) * (1 - np.power(h / 100, 8))
        kw = kl * (.581 * np.exp(.0365 * t))
        m = ed + (mo - ed) / np.power(10, kw)
    else:
        ew = .618 * np.power(h, .753) \
             + (10 * np.exp((h - 100) / 10)) \
             + .18 * (21.1 - t) * (1 - 1 / np.exp(.115 * h))

        if mo < ew:
            kl = .424 * (1 - np.power((100 - h) / 100, 1.7)) \
                 + (.0694 * np.sqrt(w)) * (1 - np.power((100 - h) / 100, 8))
            kw = kl * (.581 * np.exp(.0365 * t))
            m = ew - (ew - mo) / np.power(10, kw)
        else:
            m = mo

    ffmc = (59.5 * (250 - m)) / (147.2 + m)

    if ffmc > 101:
        ffmc = 101

    if ffmc < 0:
        ffmc = 0

    return ffmc


def compute_dmc(t, h, p, month, dmc0):
    """Compute DMC (Duff Moisture Code) given yesterday's DMC value 'dmc0',
     temperature, relative humidity, precipitation, and month"""

    dl = [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6]

    if t < -1.1:
        t = -1.1

    rk = 1.894 * (t + 1.1) * (100 - h) * (dl[int(month) - 1] * 1e-4)

    if p > 1.5:
        ra = p

        rw = .92 * ra - 1.27
        wmi = 20 + 280 / np.exp(.023 * dmc0)

        if dmc0 > 33:
            if dmc0 > 65:
                b = 6.2 * np.log(dmc0) - 17.2
            else:
                b = 14 - 1.3 * np.log(dmc0)
        else:
            b = 100 / (.5 + .3 * dmc0)

        wmr = wmi + (1e3 * rw) / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - np.log(wmr - 20))
    else:
        pr = dmc0

    if pr < 0:
        pr = 0

    dmc = pr + rk

    if dmc < 1:
        dmc = 1

    return dmc


def compute_dc(t, p, month, dc0):
    """Compute DC (Drought Code) given yesterday's DC value 'dc0',
    temperature, precipitation, and month"""

    fl = [-1.6, -1.6, -1.6, .9, 3.8, 5.8, 6.4, 5.0, 2.4, .4, -1.6, -1.6]

    if t < -2.8:
        t = -2.8

    pe = (0.36 * (t + 2.8) + fl[int(month) - 1]) / 2
    if pe < 0:
        pe = 0

    dc = pe

    if p > 2.8:
        ra = p
        rw = .83 * ra - 1.27
        smi = 800 * np.exp(-dc0 / 400)
        dr = dc0 - 400 * np.log(1 + ((3.937 * rw) / smi))

        if dr > 0:
            dc += dr
    else:
        dc += dc0

    return dc


def compute_isi(wind, ffmc):
    """Compute ISI (Initial Spread Index) given FFMC value 'ffmc' and wind"""

    mo = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    ff = 19.115 * np.exp(mo * -.1386) * (1 + np.power(mo, 5.31) / 493e5)
    isi = ff * np.exp(.05039 * wind)

    return isi


def compute_bui(dmc, dc):
    """Compute BUI (Buildup Index) given DMC 'dmc' and DC 'dc' values"""

    bui = np.where(dmc > .4,
                   dmc - (1 - .8 * dc / (dmc + 0.4 * dc)) * (.92 + np.power(.0114 * dmc, 1.7)),
                   (.8 * dc * dmc) / (dmc + .4 * dc))

    bui = np.where(bui < 0, 0, bui)

    return bui


def compute_fwi(isi, bui):
    """Compute fwi.py (Fire Weather Index) given ISI 'isi' and BUI 'bui' values"""

    bb = np.where(bui > 80,
                  .1 * isi * (1e3 / (25 + 108.64 / np.exp(.023 * bui))),
                  .1 * isi * (.626 * np.power(bui, .809) + 2))

    fwi = np.where(bb > 1,
                   np.exp(2.72 * np.power(.434 * np.log(bb), .647)),
                   bb)

    return fwi

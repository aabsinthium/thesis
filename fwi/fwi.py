import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, data, defaults):
        self.defs = defaults
        self.data = pd.DataFrame(
            data=data,
            columns=['temp', 'humid', 'wind', 'precip', 'month'])

        self.i = len(self.data) - 1
        self.codes = []

    def get_codes_df(self, i):
        data_df = self.data
        comp = Computation(*data_df.iloc[i].values)

        if i != 0:
            codes = comp.get_codes(
                *self.get_codes_df(i - 1)
            )
        else:
            codes = comp.get_codes(self.defs['ffmc'],
                                   self.defs['dmc'],
                                   self.defs['dc'])

        self.codes.append(codes)
        return codes

    def get_fwi_series(self):
        comp = Computation
        self.get_codes_df(self.i)
        codes_df = pd.DataFrame(self.codes, columns=['ffmc', 'dmc', 'dc'])
        fwi = pd.Series(
            comp.get_fwi(
                wind=self.data['wind'], ffmc=codes_df['ffmc'], dmc=codes_df['dmc'], dc=codes_df['dc']))
        return fwi


class Computation:
    def __init__(self, temp, humid, wind, precip, month):
        self.t = temp
        self.h = humid
        self.w = wind
        self.p = precip
        self.m = month

    def compute_ffmc(self, ffmc0):
        """Compute FFMC (Fine Fuel Moisture Code) given yesterday's FFMC value 'ffmc0',
         temperature, relative humidity, wind, and precipitation"""

        mo = (147.2 * (101 - ffmc0)) / (59.5 + ffmc0)
        t = self.t
        h = self.h
        w = self.w
        p = self.p

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

    def compute_dmc(self, dmc0):
        """Compute DMC (Duff Moisture Code) given yesterday's DMC value 'dmc0',
         temperature, relative humidity, precipitation, and month"""

        el = [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6]
        t = self.t
        h = self.h
        p = self.p
        month = self.m

        #els = np.array(map(lambda m: el[m-1], month))
        if t < -1.1:
            t = -1.1

        rk = 1.894 * (t + 1.1) * (100 - h) * (el[int(month) - 1] * 1e-4)

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

    def compute_dc(self, dc0):
        """Compute DC (Drought Code) given yesterday's DC value 'dc0',
        temperature, precipitation, and month"""

        fl = [-1.6, -1.6, -1.6, .9, 3.8, 5.8, 6.4, 5.0, 2.4, .4, -1.6, -1.6]
        t = self.t
        p = self.p
        month = self.m

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

    @staticmethod
    def compute_isi(wind, ffmc):
        """Compute ISI (Initial Spread Index) given FFMC value 'ffmc' and wind"""

        mo = 147.2 * (101 - ffmc) / (59.5 + ffmc)
        ff = 19.115 * np.exp(mo * -.1386) * (1 + np.power(mo, 5.31) / 493e5)
        isi = ff * np.exp(.05039 * wind)

        return isi

    @staticmethod
    def compute_bui(dmc, dc):
        """Compute BUI (Buildup Index) given DMC 'dmc' and DC 'dc' values"""

        bui = np.where(dmc > .4,
                       dmc - (1 - .8 * dc / (dmc + 0.4 * dc)) * (.92 + np.power(.0114 * dmc, 1.7)),
                       (.8 * dc * dmc) / (dmc + .4 * dc))

        bui = np.where(bui < 0, 0, bui)

        return bui

    @staticmethod
    def compute_fwi(isi, bui):
        """Compute fwi.py (Fire Weather Index) given ISI 'isi' and BUI 'bui' values"""

        bb = np.where(bui > 80,
                      .1 * isi * (1e3 / (25 + 108.64 / np.exp(.023 * bui))),
                      .1 * isi * (.626 * np.power(bui, .809) + 2))

        fwi = np.where(bb > 1,
                       np.exp(2.72 * np.power(.434 * np.log(bb), .647)),
                       bb)

        return fwi

    def get_codes(self, ffmc0, dmc0, dc0):
        ffmc = self.compute_ffmc(ffmc0)
        dmc = self.compute_dmc(dmc0)
        dc = self.compute_dc(dc0)

        return [ffmc, dmc, dc]

    @staticmethod
    def get_fwi(wind, ffmc, dmc, dc):
        isi = Computation.compute_isi(wind, ffmc)
        bui = Computation.compute_bui(dmc, dc)

        fwi = Computation.compute_fwi(isi, bui)

        return fwi

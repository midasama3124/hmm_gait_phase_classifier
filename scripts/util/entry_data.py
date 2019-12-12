import collections
import math
import string
import numpy as np
import statistics as st
from collections import namedtuple

DataEntry = namedtuple('DataEntry', \
                       'quatx quaty quatz quatw \
        gyrox gyroy gyroz \
        accelx accely accelz \
        compx compy compz \
        label \
        sequence'
                       )


class fullEntry:
    # def __init__(self, quatx, quaty, quatz, quatw,\
    #        gyrox, gyroy, gyroz,\
    #        accelx, accely, accelz,\
    #        compx, compy, compz,
    #        label,
    #        sequence):
    def __init__(self):
        self.features = []
        self.labels = []
        self.seq = []

    def add(self, entry):
        feat = [entry.quatx]
        feat.append(entry.quaty)
        feat.append(entry.quatz)
        feat.append(entry.quatw)
        feat.append(entry.gyrox)
        feat.append(entry.gyroy)
        feat.append(entry.gyroz)
        feat.append(entry.accelx)
        feat.append(entry.accely)
        feat.append(entry.accelz)
        feat.append(entry.compx)
        feat.append(entry.compy)
        feat.append(entry.compz)
        self.features.append(feat)
        self.labels.append(entry.label)
        self.seq.append(entry.sequence)

    def len(self):
        return len(self.features)


class dataParams:
    def __init__(self, **kwargs):
        self.data = kwargs.get('data', [])
        self.label = kwargs.get('label', -1)
        self.min = 0.0
        self.max = 0.0
        self.mean = 0.0
        self.stdev = 0.0
        self.variance = 0.0

    def calcParams(self):
        self.min = min(self.data)
        self.max = max(self.data)
        self.mean = sum(self.data) / float(len(self.data))
        self.stdev = st.stdev(self.data)
        self.variance = st.variance(self.data)


class classData:
    def __init__(self, **kwargs):
        self.quatx = dataParams(data=kwargs.get('quatx', []), label=kwargs.get('label', -1))
        self.quaty = dataParams(data=kwargs.get('quaty', []), label=kwargs.get('label', -1))
        self.quatz = dataParams(data=kwargs.get('quatz', []), label=kwargs.get('label', -1))
        self.quatw = dataParams(data=kwargs.get('quatw', []), label=kwargs.get('label', -1))
        self.gyrox = dataParams(data=kwargs.get('gyrox', []), label=kwargs.get('label', -1))
        self.gyroy = dataParams(data=kwargs.get('gyroy', []), label=kwargs.get('label', -1))
        self.gyroz = dataParams(data=kwargs.get('gyroz', []), label=kwargs.get('label', -1))
        self.accelx = dataParams(data=kwargs.get('accelx', []), label=kwargs.get('label', -1))
        self.accely = dataParams(data=kwargs.get('accely', []), label=kwargs.get('label', -1))
        self.accelz = dataParams(data=kwargs.get('accelz', []), label=kwargs.get('label', -1))
        self.compx = dataParams(data=kwargs.get('compx', []), label=kwargs.get('label', -1))
        self.compy = dataParams(data=kwargs.get('compy', []), label=kwargs.get('label', -1))
        self.compz = dataParams(data=kwargs.get('compz', []), label=kwargs.get('label', -1))

    def add(self, entry):
        self.quatx.data.append(entry.quatx)
        self.quaty.data.append(entry.quaty)
        self.quatz.data.append(entry.quatz)
        self.quatw.data.append(entry.quatw)
        self.gyrox.data.append(entry.gyrox)
        self.gyroy.data.append(entry.gyroy)
        self.gyroz.data.append(entry.gyroz)
        self.accelx.data.append(entry.accelx)
        self.accely.data.append(entry.accely)
        self.accelz.data.append(entry.accelz)
        self.compx.data.append(entry.compx)
        self.compy.data.append(entry.compy)
        self.compz.data.append(entry.compz)

    def calcParams(self):
        self.quatx.calcParams()
        self.quaty.calcParams()
        self.quatz.calcParams()
        self.quatw.calcParams()
        self.gyrox.calcParams()
        self.gyroy.calcParams()
        self.gyroz.calcParams()
        # self.accelx.calcParams()
        # self.accely.calcParams()
        # self.accelz.calcParams()
        # self.compx.calcParams()
        # self.compy.calcParams()
        # self.compz.calcParams()

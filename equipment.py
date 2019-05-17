import numpy as np
from enum import Enum
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

import room

class HeatRegulationEquipment():
    def __init__(self, name='', max_power=0, is_on=False, heater=True, dt=0):
        self.name = name
        self.max_power = max_power
        #self.is_on = is_on
        self.heater = heater
        self.timestep = dt
        self.total_power_used = 0
        self.output = 0
    
    def regulate_output(self, message):
        pass

    def produce(self):
        self.total_power_used += self.output * self.timestep
        return self.output

class Heater(HeatRegulationEquipment):
    def __init__(self, name='', power=0, is_on=False):
        super().__init__(name, power, is_on, heater=True)

class Cooler(HeatRegulationEquipment):
    def __init__(self, name='', power=0, is_on=False):
        super().__init__(name, power, is_on, heater=False)

class TemperatureSensor():
    def __init__(self, name, brand, temperature = 0):
        self.name = name
        #self.room = room
        self.temperature = temperature
        self.brand = brand

    def measure_temperature(self, room):
        self.temperature = room.temperature + np.random.randn() * 0.1

    def temperature_service(self):
        return {}

if __name__ == '__main__':
    pass

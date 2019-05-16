import numpy as np
from enum import Enum
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

import room

class HeatRegulationEquipment():
    def __init__(self, name='', power=0, is_on=False, heater=True):
        self.name = name
        self.power = power
        self.is_on = is_on
        self.heater = heater
        if is_on and heater:
            self.output = power
        elif is_on and not heater:
            self.output = -power
        elif not is_on:
            self.output = 0
        self.regulation = 0
        self.total_power_used = 0
        self.integrator = 0
    
    def turn_on(self):
        self.is_on = True

    def turn_off(self):
        self.is_on = False
        self.output = 0

    def regulate(self, current_temperature, requested_temperature):
        self.regulation = 0.1 * (requested_temperature - current_temperature)
        self.integrator += 0.0001 * (requested_temperature - current_temperature) * 1
        if self.integrator > 1:
            self.integrator = 1
        elif self.integrator < -1:
            self.integrator = -1

        if self.heater:
            self.output = min(self.power, max(0, self.power * self.regulation + self.integrator))
            #self.output = self.power * self.regulation
        else:
            # I think this line is bugged, check if he the output is always at -self.power or if it actually changes. From what I can see it looks like it should always give -self.power
            self.output = max(-self.power, min(0, -self.power * -self.regulation + self.integrator))
            #self.output = -self.power * self.regulation

    def produce(self):
        self.total_power_used += 1*self.output
        return self.output

class Heater(HeatRegulationEquipment):
    def __init__(self, name='', power=0, is_on=False):
        super().__init__(name, power, is_on, heater=True)

class Cooler(HeatRegulationEquipment):
    def __init__(self, name='', power=0, is_on=False):
        super().__init__(name, power, is_on, heater=False)

class TemperatureSensor():
    def __init__(self, name, room, brand):
        self.name = name
        self.room = room
        self.brand = brand

    def measure_temperature(self):
        return self.room.temperature + np.random.randn() * 0.5

    def temperature_service(self):
        return {}

if __name__ == '__main__':
    test_heater = Heater('test_heater', 100)
    test_cooler = Cooler('test_cooler', 200)
    print(test_heater.produce())
    print(test_cooler.produce())
    test_heater.turn_on()
    test_cooler.turn_on()
    print(test_heater.produce())
    print(test_cooler.produce())

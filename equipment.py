import numpy as np
from enum import Enum

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
    
    def turn_on(self):
        self.is_on = True
        if self.heater:
            self.output = self.power
        else:
            self.output = -self.power

    def turn_off(self):
        self.is_on = True
        self.output = 0

    def produce(self):
        return self.output

"""
    Dunno what this superclass would be good for but I'll keep it for now
    """

class Heater(HeatRegulationEquipment):
    def __init__(self, name='', power=0, is_on=False):
        super().__init__(name, power, is_on, heater=True)

class Cooler(HeatRegulationEquipment):
    def __init__(self, name='', power=0, is_on=False):
        super().__init__(name, power, is_on, heater=False)

if __name__ == '__main__':
    test_heater = Heater('test_heater', 100)
    test_cooler = Cooler('test_cooler', 200)
    print(test_heater.produce())
    print(test_cooler.produce())
    test_heater.turn_on()
    test_cooler.turn_on()
    print(test_heater.produce())
    print(test_cooler.produce())

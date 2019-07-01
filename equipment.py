import numpy as np
from enum import Enum
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

import room

class HeatRegulationEquipment():
    def __init__(self, name='', max_power=0, is_on=False, heater=True, coordinates=False, dt=0):
        self.name = name
        self.max_power = max_power
        #self.is_on = is_on
        self.heater = heater
        self.timestep = dt
        self.coordinates = coordinates
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
    def __init__(self, name, brand, temperature = 0, coordinates=False):
        self.name = name
        self.temperature = temperature
        self.brand = brand
        self.coordinates = coordinates

    def measure_temperature(self, room):
        self.temperature = room.temperature + np.random.randn() * 0.05

    def temperature_service(self):
        return {}

class Controller():
    def __init__(self, setpoint,
            heater_id='', cooler_id='', indoor_temp_sensor_id='', outdoor_temp_sensor_id='', coordinates=False):
        self.setpoint = setpoint
        self.temperature = 0
        self.heater_id = heater_id
        self.cooler_id = cooler_id
        self.indoor_temp_sensor_id = indoor_temp_sensor_id
        self.outdoor_temp_sensor_id = outdoor_temp_sensor_id
        self.coordinates = coordinates

    def update_setpoint(self, new_setpoint):
        """ Updates the setpoint, perhaps a pointless method """
        self.setpoint = new_setpoint

    def read_temperature(self, temperature_message):
        """ Receives a SenML message and updates the current temperature """
        pass

    def update_control(self):
        """ Updates the control value """
        pass

    def read_and_update(self, temperature_message):
        """ Reads temperature and updates control value """
        pass

    def heater_message(self):
        """ Returns heater message """
        pass

    def cooler_message(self):
        """ Returns cooler message """
        pass

if __name__ == '__main__':
    pass

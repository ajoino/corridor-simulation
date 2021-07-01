import numpy as np
import equipment
#from equipment import equipment.Heater, equipment.Cooler
from collections import namedtuple

#HeatControlSystem = namedtuple('Heat Control System', ['actuator', 'controller', 'sensor'])

class Room():
    def __init__(self, name='', temperature=0, heat_capacity=1, 
            heater_temperature_sensor=None, cooler_temperature_sensor=None, position=None, static=False):
        self.name = name
        self.temperature = temperature
        self.new_temperature = temperature
        #self.requested_temperature = requested_temperature
        self.neighbors = []
        self.static = static
        self.heater_temperature_sensor=heater_temperature_sensor
        self.cooler_temperature_sensor=cooler_temperature_sensor
        self.heat_capacity = heat_capacity
        self.position = position

    def __repr__(self):
        return f'Room: {self.name}\n\tTemperature: {self.temperature}\n'

    def __str__(self):
        #return self.__repr__()
        return f'Room {self.name}'

    def change_temperature(self, new_temp):
        self.temperature = new_temp

    def update_temperature(self, dt, coefficient):
        if self.static:
            return
        neighbor_temp_diff = [self.temperature - neighbor.temperature for neighbor in self.neighbors]

        self.new_temperature = self.temperature - dt * coefficient * sum(neighbor_temp_diff)

    def execute_temperature(self):
        if self.static:
            self.heater_temperature_sensor.measure_temperature(self)
            self.cooler_temperature_sensor.measure_temperature(self)
            return
        self.temperature = self.new_temperature
        self.heater_temperature_sensor.measure_temperature(self)
        self.cooler_temperature_sensor.measure_temperature(self)

    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            if not isinstance(neighbor, Room):
                print(f'\n{neighbor} is not of class Room')
                continue
            if neighbor in self.neighbors:
                #print(f'\n{neighbor} is already a neighbor of {self}', file=sys.stderr)
                continue
            if neighbor is self:
                print(f'\n{self} cannot be its own neighbor')
                continue
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)

class Office(Room):
    def __init__(self, name='', temperature=None, heat_capacity=1, 
            heater_temperature_sensor=None, cooler_temperature_sensor=None, position=None, heater=None, cooler=None, heater_controller=None, cooler_controller=None):
        super().__init__(name, temperature, heat_capacity, heater_temperature_sensor=heater_temperature_sensor, cooler_temperature_sensor=cooler_temperature_sensor)
        self.heater = heater
        self.cooler = cooler
        self.heater_controller = heater_controller
        self.cooler_controller = cooler_controller

    def __str__(self):
        return f'Office {self.name}'

    def update_temperature(self, dt, coefficient):
        super().update_temperature(dt, coefficient)

        self.new_temperature += dt / self.heat_capacity * (self.heater.produce() + self.cooler.produce())

    def update_control(self, dt, time):
        heater_temperature_message = self.heater_temperature_sensor.temperature_service(time)
        cooler_temperature_message = self.cooler_temperature_sensor.temperature_service(time)
        self.heater_controller.read_and_update(heater_temperature_message, dt, time)
        self.cooler_controller.read_and_update(cooler_temperature_message, dt, time)
        heater_message = self.heater_controller.heater_message(time)
        cooler_message = self.cooler_controller.cooler_message(time)
        self.heater.regulate_output(heater_message)
        self.cooler.regulate_output(cooler_message)
        return [heater_temperature_message, heater_message, cooler_temperature_message, cooler_message]

class Outside(Room):
    pass

if __name__ == '__main__':
    pass

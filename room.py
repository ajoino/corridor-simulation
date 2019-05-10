import numpy as np
from equipment import Heater, Cooler

class Room():
    def __init__(self, name='', temperature=0, requested_temperature=0, position=None, static=False):
        self.name = name
        self.temperature = temperature
        self.new_temperature = temperature
        self.requested_temperature = requested_temperature
        self.neighbors = []
        self.static = static

    def __repr__(self):
        return f'Room: {self.name}\n\tTemperature: {self.temperature}\n'

    def __str__(self):
        #return self.__repr__()
        return f'Room {self.name}'

    def update_temperature(self, dt, coefficient):
        #self.temperature = new_temperature
        if self.static:
            return
        neighbor_temperatures = np.array([neighbor.temperature for neighbor in self.neighbors])
        extra_temp = 0
        if hasattr(self, 'heater'):
            extra_temp = dt * (self.heater.produce() + self.cooler.produce())

        self.new_temperature = self.temperature - dt * coefficient * np.sum(self.temperature - neighbor_temperatures) + extra_temp
        #return new_temperature

    def execute_temperature(self):
        self.temperature = self.new_temperature

    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            if not isinstance(neighbor, Room):
                print(f'\n{neighbor} is not of class Room')
                continue
            if neighbor in self.neighbors:
                print(f'\n{neighbor} is already a neighbor of {self}')
                continue
            if neighbor is self:
                print(f'\n{self} cannot be its own neighbor')
                continue
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)

class Office(Room):
    def __init__(self, name='', temperature=None, requested_temperature=None, position=None, heater_power=0, cooler_power=0):
        super().__init__(name, temperature, requested_temperature)
        self.heater = Heater(self.name, heater_power)
        self.cooler = Cooler(self.name, cooler_power)

    def __str__(self):
        return f'Office {self.name}'

class Outside(Room):
    pass

if __name__ == '__main__':
    test_room = Room('test', 0)
    print(test_room)
    test_room.update_temperature(273)
    print(test_room)
    static_room = Room('static', 298, True)
    print(static_room)
    static_room.update_temperature(273)
    print(static_room)

    test_room.add_neighbors([static_room])
    print(test_room.neighbors)
    print(static_room.neighbors)
    test_room.add_neighbors([static_room])
    test_room.add_neighbors([test_room])
    
    test_office = Office('Office', 295)
    print(test_office)

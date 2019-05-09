import numpy as np
from equipment import Heater, Cooler

class Room():
    def __init__(self, name='', temperature=None, static_temperature=False, position=None):
        self.name = name
        self.temperature = temperature
        self.static_temperature = static_temperature
        self.neighbors = []

    def __repr__(self):
        return f'Room: {self.name}\n\tTemperature: {self.temperature}'

    def __str__(self):
        #return self.__repr__()
        return f'Room {self.name}'

    def update_temperature(self, new_temperature):
        if self.static_temperature:
            return None
        
        self.temperature = new_temperature
        return new_temperature

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
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)

class Office(Room):
    def __init__(self, name='', temperature=None, position=None, heater_power, cooler_power):
        super().__init__(name, temperature, static_temperature=False)
        heater = Heater(self.name, heater_power)
        cooler = Cooler(self.name, cooler_power)

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

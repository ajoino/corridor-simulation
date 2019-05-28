import numpy as np
import equipment
#from equipment import equipment.Heater, equipment.Cooler

class Room():
    def __init__(self, name='', temperature=0, heat_capacity=1, 
            temperature_sensor=None, position=None, static=False):
        self.name = name
        self.temperature = temperature
        self.new_temperature = temperature
        #self.requested_temperature = requested_temperature
        self.neighbors = []
        self.static = static
        self.temperature_sensor=temperature_sensor
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
        #self.temperature = new_temperature
        if self.static:
            return
        neighbor_temperatures = np.array([neighbor.temperature for neighbor in self.neighbors])
        extra_temp = 0
        if hasattr(self, 'heater'):
            extra_temp = dt / self.heat_capacity * (self.heater.produce() + self.cooler.produce())

        self.new_temperature = self.temperature - dt * coefficient * np.sum(self.temperature - neighbor_temperatures) + extra_temp
        #return new_temperature

    def execute_temperature(self):
        if self.static:
            return
        self.temperature = self.new_temperature
        self.temperature_sensor.measure_temperature(self)

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
            temperature_sensor=None, position=None, heater=None, cooler=None, controller=None):
        super().__init__(name, temperature, heat_capacity, temperature_sensor)
        self.heater = heater
        self.cooler = cooler
        self.controller = controller

    def __str__(self):
        return f'Office {self.name}'

    def update_control(self, dt, time):
        temperature_message = self.temperature_sensor.temperature_service(time)
        self.controller.read_and_update(temperature_message, dt, time)
        heater_message = self.controller.heater_message(time)
        cooler_message = self.controller.cooler_message(time)
        self.heater.regulate_output(heater_message)
        self.cooler.regulate_output(cooler_message)
        return temperature_message, heater_message, cooler_message

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

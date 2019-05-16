import equipment
import room
import json

class AsysHeater(equipment.Heater):
    def __init__(self, name='', power=0):
        super().__init__(name, power, heater=True)

    def heat_service_provider(self, message):
        pass

class AsysCooler(equipment.Cooler):
    def __init__(self, name='', power=0):
        super().__init__(name, power, heater=True)

class AsysTemperatureSensor(equipment.TemperatureSensor):
    def __init__(self, name, room):
        super().__init__(name, room, 'Asys')

    def temperature_service(self):
        message = [{'n': self.name, 't': 'timestamp', 'u': 'Cel', 'v': self.measure_temperature()}]
        return json.dumps(message)

class AsysController():
    pass

if __name__ == '__main__':
    office = room.Office('office', 293)
    temp_sensor = AsysTemperatureSensor('office', office)
    print(temp_sensor.temperature_service())

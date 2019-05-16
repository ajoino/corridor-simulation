import equipment
import room
import json

class BtechTemperatureSensor(equipment.TemperatureSensor):
    def __init__(self, name, room):
        super().__init__(name, room, 'B Tech')

    def temperature_service(self):
        message = [{'bn': self.name, 'bt': 'timestamp', 'u': 'K', 'v': self.measure_temperature() + 273.15},
                   {'u': 'lon', 'v': 120},
                   {'u': 'lat', 'v': 30}]
        return json.dumps(message)

if __name__ == '__main__':
    office = room.Office('office', 293)
    temp_sensor = BtechTemperatureSensor('office', office)
    print(temp_sensor.temperature_service())

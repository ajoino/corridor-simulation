import numpy as np
import equipment
import room
from utils import celsius

class PowerOutput:
    def __set_name__(self, owner, name):
        self.instance_name = '_' + name

    def __init__(self, max_power: float):
        self.max_power = max_power

    def __get__(self, instance, owner):
        if instance is None:
            return self

        return instance._power

    def __set__(self, instance, value):
        self._power = max(0, min(value, self.max_power))

class Heater(equipment.HeatRegulationEquipment):
    """ Heater of brand B-tech """
    def __init__(self, name='', max_power=0, coordinates=0):
        super().__init__(name, max_power, coordinates=coordinates)

    def regulate_output(self, message):
        """ Receives a SenML message and updates the output of the heater """
        metadata, measurement, longitude, latitude = message
        if metadata['bn'] != self.name:
            return
        elif longitude['u'] != 'Lon' and longitude['v'] != self.coordinates[0]:
            return
        elif latitude['u'] != 'Lat' and latitude['v'] != self.coordinates[1]:
            return
        elif measurement['u'] != '/':
            return
        self.output = self.max_power * np.clip(measurement['v'], 0, 1)

class Cooler(equipment.HeatRegulationEquipment):
    """ Cooler of brand B-tech """
    def __init__(self, name='', power=0, coordinates=0):
        super().__init__(name, power, coordinates=coordinates)

    def regulate_output(self, message):
        """ Receives a SenML message and updates the output of the cooler """
        metadata, measurement, longitude, latitude = message
        if metadata['bn'] != self.name:
            return
        elif longitude['u'] != 'Lon' and longitude['v'] != self.coordinates[0]:
            return
        elif latitude['u'] != 'Lat' and latitude['v'] != self.coordinates[1]:
            return
        elif measurement['u'] != '/':
            return
        self.output = min(-self.max_power * measurement['v'], 0)

class TemperatureSensor(equipment.TemperatureSensor):
    """ Temperature sensor of brand B-tech """
    def __init__(self, name, temperature, coordinates=0):
        super().__init__(name, 'B-tech', temperature, coordinates=coordinates)

    def temperature_service(self, timestep):
        """ Returns a SenML message containing the temperature of the room """
        message = [{'bn': self.name, 'bt': int(timestep)},
                   {'u': 'Cel', 'v': celsius(self.temperature)},
                   {'u': 'Lon', 'v': f'{self.coordinates[0]}'},
                   {'u': 'Lat', 'v': f'{self.coordinates[1]}'}]
        return message

class Controller(equipment.Controller):
    """ PI Controller of brand B-tech """
    def __init__(self, setpoint, k_P, k_I, 
            heater_id='', cooler_id='', indoor_temp_sensor_id='', outdoor_temp_sensor_id='', coordinates=0):
        super().__init__(setpoint, heater_id, cooler_id, indoor_temp_sensor_id, outdoor_temp_sensor_id, coordinates=coordinates)
        self.integral = 0
        self.k_P = k_P
        self.k_I = k_I
        self.control = 0

    def read_temperature(self, temperature_message):
        """ Receives a SenML message and updates the current temperature """
        metadata, measurement, longitude, latitude = temperature_message
        if metadata['bn'] != self.indoor_temp_sensor_id:
            return
        elif longitude['u'] != 'Lon' and longitude['v'] != self.coordinates[0]:
            return
        elif latitude['u'] != 'Lat' and latitude['v'] != self.coordinates[1]:
            return
        elif measurement['u'] != 'Cel':
            return
        self.temperature = measurement['v']

    def update_control(self, dt, P=True, I=True, D=True):
        """ Updates the control value """
        error = self.setpoint - self.temperature
        self.integral = max(-10 * self.k_P, min(self.integral + error * dt, 10 * self.k_P))
        self.control = max(-1, min(self.k_P * error + self.k_I * self.integral, 1))

    def read_and_update(self, temperature_message, dt, P=True, I=True, D=True):
        """ Reads temperature and updates control value """
        self.read_temperature(temperature_message)
        self.update_control(dt, P, I, D)

    def heater_message(self, timestep):
        """ Returns heater message """
        message = [{'bn': self.heater_id, 'bt': int(timestep)}, {'u': '/', 'v': self.control}, {'u': 'Lon', 'v': f'{self.coordinates[0]}'}, {'u': 'Lat', 'v': f'{self.coordinates[1]}'}]
        return message

    def cooler_message(self, timestep):
        """ Returns cooler message """
        message = [{'bn': self.cooler_id, 'bt': int(timestep)}, {'u': '/', 'v': -min(self.control, 0)}, {'u': 'Lon', 'v': f'{self.coordinates[0]}'}, {'u': 'Lat', 'v': f'{self.coordinates[1]}'}]
        return message
        #return f"""[{{"n": "{self.cooler_id}", "t": "timestamp", "u": "W", "v": {-max(0, self.control)}}}]"""

if __name__ == '__main__':
    exit()
    np.random.seed(1337)
    office = room.Office('office', 293.5)
    temp_sensor = TemperatureSensor('office', office)
    heater = Heater('office_heater', 1500)
    cooler = Cooler('office_cooler', 1000)
    print(temp_sensor.temperature_service())
    controller = Controller(293, 0.1, 0.1,
            heater.name, cooler.name, temp_sensor.name)
    controller.read_temperature(temp_sensor.temperature_service())
    controller.update_control(10)
    print(controller.heater_message())
    print(controller.cooler_message())
    print(controller.integral)
    for i in range(200):
        controller.update_control(10)
    print(controller.heater_message())
    print(controller.cooler_message())
    print(controller.integral)

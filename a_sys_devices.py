import numpy as np
import equipment
import room
import json

class Heater(equipment.HeatRegulationEquipment):
    """ Heater of brand Asys """
    def __init__(self, name='', max_power=0):
        super().__init__(name, max_power)

    def regulate_output(self, message):
        """ Receives a SenML message and updates the output of the heater """
        measurement = json.loads(message)[0]
        if measurement['n'] != self.name:
            return
        elif measurement['u'] != 'W':
            return
        self.output = np.clip(measurement['v'], 0, self.max_power)

class Cooler(equipment.HeatRegulationEquipment):
    """ Cooler of brand Asys """
    def __init__(self, name='', power=0):
        super().__init__(name, power)

    def regulate_output(self, message):
        """ Receives a SenML message and updates the output of the cooler """
        measurement = json.loads(message)[0]
        if measurement['n'] != self.name:
            return
        elif measurement['u'] != 'W':
            return
        self.output = -np.clip(measurement['v'], 0, self.max_power)

class TemperatureSensor(equipment.TemperatureSensor):
    """ Temperature sensor of brand Asys """
    def __init__(self, name, temperature):
        super().__init__(name, 'Asys', temperature)

    def temperature_service(self, time):
        """ Returns a SenML message containing the temperature of the room """
        message = [{'n': self.name, 't': int(time), 'u': 'K', 'v': self.temperature}]
        return json.dumps(message)

class Controller(equipment.Controller):
    """ PI Controller of brand Asys """
    def __init__(self, setpoint, k_P, k_I, 
            heater_id='', cooler_id='', indoor_temp_sensor_id='', outdoor_temp_sensor_id=''):
        super().__init__(setpoint, heater_id, cooler_id, indoor_temp_sensor_id, outdoor_temp_sensor_id)
        self.integral = 0
        self.k_P = k_P
        self.k_I = k_I
        self.control = 0

    def read_temperature(self, temperature_message):
        """ Receives a SenML message and updates the current temperature """
        message = json.loads(temperature_message)
        measurement = message[0]
        if measurement['n'] != self.indoor_temp_sensor_id:
            return
        elif measurement['u'] != 'K':
            return
        self.temperature = measurement['v']

    def update_control(self, dt, P=True, I=True, D=True):
        """ Updates the control value """
        error = self.setpoint - self.temperature
        self.integral = np.clip(self.integral + error * dt, -10 * self.k_P, 10 * self.k_P)
        self.control = self.k_P * error + self.k_I * self.integral

    def read_and_update(self, temperature_message, dt, P=True, I=True, D=True):
        """ Reads temperature and updates control value """
        self.read_temperature(temperature_message)
        self.update_control(dt, P, I, D)

    def heater_message(self, time):
        """ Returns heater message """
        message = [{'n': self.heater_id, 't': int(time), 'u': 'W', 'v': max(0, self.control)}]
        return json.dumps(message)
        #return f"""[{{"n": "{self.heater_id}", "t": "timestamp", "u": "W", "v": {max(0, self.control)}}}]"""

    def cooler_message(self, time):
        """ Returns cooler message """
        message = [{'n': self.cooler_id, 't': int(time), 'u': 'W', 'v': -min(self.control, 0)}]
        return json.dumps(message)
        #return f"""[{{"n": "{self.cooler_id}", "t": "timestamp", "u": "W", "v": {-max(0, self.control)}}}]"""

if __name__ == '__main__':
    np.random.seed(1337)
    office = room.Office('office', 293.5)
    temp_sensor = TemperatureSensor('office', office)
    heater = Heater('office_heater', 1500)
    cooler = Cooler('office_cooler', 1000)
    print(temp_sensor.temperature_service(time))
    controller = Controller(293, 0.1, 0.1,
            heater.name, cooler.name, temp_sensor.name)
    controller.read_temperature(temp_sensor.temperature_service(time))
    controller.update_control(10)
    print(controller.heater_message(time))
    print(controller.cooler_message(time))
    print(controller.integral)
    for i in range(200):
        controller.update_control(10)
    print(controller.heater_message(time))
    print(controller.cooler_message(time))
    print(controller.integral)

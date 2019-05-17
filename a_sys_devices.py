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

    def temperature_service(self):
        """ Returns a SenML message containing the temperature of the room """
        message = [{'n': self.name, 't': 'timestamp', 'u': 'Cel', 'v': self.temperature}]
        return json.dumps(message)

class Controller():
    """ PI Controller of brand Asys """
    def __init__(self, setpoint, k_P, k_I, 
            heater_id=None, cooler_id=None, indoor_temp_sensor_id='', outdoor_temp_sensor_id=''):
        self.setpoint = setpoint
        self.temperature = 0
        self.integral = 0
        self.k_P = k_P
        self.k_I = k_I
        self.control = 0
        self.heater_id = heater_id
        self.cooler_id = cooler_id
        self.indoor_temp_sensor_id = indoor_temp_sensor_id
        self.outdoor_temp_sensor_id = outdoor_temp_sensor_id

    def update_setpoint(self, new_setpoint):
        """ Updates the setpoint, perhaps a pointless method """
        self.setpoint = new_setpoint

    def read_temperature(self, temperature_message):
        """ Receives a SenML message and updates the current temperature """
        message = json.loads(temperature_message)
        measurement = message[0]
        if measurement['n'] != self.indoor_temp_sensor_id:
            return
        elif measurement['u'] != 'Cel':
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

    def heater_message(self):
        """ Returns heater message """
        message = [{'n': self.heater_id, 't': 'timestamp', 'u': 'W', 'v': max(0, self.control)}]
        return json.dumps(message)
        #return f"""[{{"n": "{self.heater_id}", "t": "timestamp", "u": "W", "v": {max(0, self.control)}}}]"""

    def cooler_message(self):
        """ Returns cooler message """
        message = [{'n': self.cooler_id, 't': 'timestamp', 'u': 'W', 'v': -max(0, self.control)}]
        return json.dumps(message)
        #return f"""[{{"n": "{self.cooler_id}", "t": "timestamp", "u": "W", "v": {-max(0, self.control)}}}]"""

if __name__ == '__main__':
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

from typing import Dict, Optional, Tuple, List

import numpy as np
from enum import Enum
import json
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.2f')

import room


class HeatRegulationEquipment():
    def __init__(
            self,
            name: str = '',
            max_power: float = 0,
            is_on: bool = False,
            heater: bool = True,
            coordinates: Optional[Tuple[float, float]] = None,
            dt: float = 0
    ):
        self.name = name
        self.max_power = max_power
        # self.is_on = is_on
        self.heater = heater
        self.timestep = dt
        self.coordinates = coordinates
        self.total_power_used = 0
        self.output = 0

    def regulate_output(self, message: List[Dict]) -> None:
        pass

    def produce(self) -> float:
        self.total_power_used += self.output * self.timestep
        return self.output


class TemperatureSensor():
    def __init__(
            self,
            name: str,
            brand: str,
            temperature: float = 0,
            coordinates: Optional[Tuple[float, float]] = None,
    ):
        self.name = name
        self.temperature = temperature
        self.brand = brand
        self.coordinates = coordinates

    def measure_temperature(self, room: room.Room) -> None:
        self.temperature = room.temperature + np.random.randn() * 0.05

    def temperature_service(self, time: float) -> List[Dict]:
        return {}


class Controller():
    def __init__(
            self,
            setpoint: float,
            heater_id: str = '',
            cooler_id: str = '',
            indoor_temp_sensor_id: str = '',
            outdoor_temp_sensor_id: str = '',
            coordinates: Optional[Tuple[float, float]] = None
    ):
        self.setpoint = setpoint
        self.temperature = 0
        self.heater_id = heater_id
        self.cooler_id = cooler_id
        self.indoor_temp_sensor_id = indoor_temp_sensor_id
        self.outdoor_temp_sensor_id = outdoor_temp_sensor_id
        self.coordinates = coordinates

    def update_setpoint(self, new_setpoint: float) -> None:
        """ Updates the setpoint, perhaps a pointless method """
        self.setpoint = new_setpoint

    def read_temperature(self, temperature_message: List[Dict]) -> None:
        """ Receives a SenML message and updates the current temperature """
        pass

    def update_control(self, dt: float) -> None:
        """ Updates the control value """
        pass

    def read_and_update(self, temperature_message: List[Dict], dt: float) -> None:
        """ Reads temperature and updates control value """
        pass

    def heater_message(self, time: float) -> List[Dict]:
        """ Returns heater message """
        pass

    def cooler_message(self, time: float) -> List[Dict]:
        """ Returns cooler message """
        pass


if __name__ == '__main__':
    pass

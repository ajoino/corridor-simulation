import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Sequence, Mapping, NamedTuple, Dict, Optional, TypedDict, Union
import time

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import RandomState
import ujson as json

import equipment
from equipment import TemperatureSensor
from room import Room
import random

class MessagePair(NamedTuple):
    a_sys: List[Dict]
    b_tech: List[Dict]

class MessageGroup(TypedDict):
    heater: Dict
    cooler: Dict

Messages = Dict[str, MessageGroup]

class SensorGroup(TypedDict):
    heater: TemperatureSensor
    cooler: TemperatureSensor

class ControllerGroup(TypedDict):
    heater: equipment.Controller
    cooler: equipment.Controller

class RegulatorGroup(TypedDict):
    heater: equipment.HeatRegulationEquipment
    cooler: equipment.HeatRegulationEquipment

class Environment(ABC):
    prng: RandomState
    outside: Room
    rooms: Dict[str, Room]
    temp_sensors: Dict[str, SensorGroup]
    controllers: Dict[str, ControllerGroup]
    heat_reg_eq: Dict[str, RegulatorGroup]

    def __init__(self):
        self.room_init()
        self.equipment_init()

    @abstractmethod
    def room_init(self):
        pass

    @abstractmethod
    def equipment_init(self):
        pass

    @abstractmethod
    def update_random_setpoints(self, current_time: float) -> Dict[str, float]:
        pass

    def read_temperature_sensors(self, current_time: float, noise_vector: Sequence[float]) -> Messages:
        for (room_name, sensors), noise in zip(self.temp_sensors.items(), noise_vector):
            sensors['heater'].temperature = self.rooms[room_name].temperature + noise
            sensors['cooler'].temperature = self.rooms[room_name].temperature + noise
        messages = {
            room_name: {target: sensor.temperature_service(current_time)
                        for target, sensor in sensors.items()}
            for room_name, sensors in self.temp_sensors.items()
        }
        return messages

    def update_control(
            self,
            temperature_messages: Messages,
            time_delta: float,
            current_time: float,
    ) -> Messages:
        for room_name, messages in temperature_messages.items():
            for target, message in messages.items():
                self.controllers[room_name][target].read_and_update(message, time_delta)
        messages = {
            room_name: {target: getattr(controller, target + '_message')(current_time)
                        for target, controller in controllers.items()}
            for room_name, controllers in self.controllers.items()
        }

        return messages

    def update_heat_regulation(self, controller_messages: Messages, time_delta: float) -> Dict[str, float]:
        for room_name, messages in controller_messages.items():
            for target, message in messages.items():
                self.heat_reg_eq[room_name][target].regulate_output(message)
        outputs = {
            room_name: equipment['heater'].output + equipment['cooler'].output
            for room_name, equipment in self.heat_reg_eq.items()
        }
        return outputs

    def update_room_temperatures(
            self,
            previous_room_temperatures: Mapping[str, float],
            current_heat_output: Mapping[str, float],
            time_delta: float):
        for room_name, output in current_heat_output.items():
            self.rooms[room_name].update_temperature(
                    previous_room_temperatures, time_delta, 1e-4, output
            )

    def room_simulation_startup(self, start_temperature: float):
        self.outside.static_temp = start_temperature + 273.15

        for room in self.rooms.values():
                room.temperature = start_temperature + 273.15

    def step(
            self,
            dt: float,
            current_time: float,
            outside_temperature: float,
            noise_vector: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, Messages, Messages]:
        setpoints = self.update_random_setpoints(current_time)
        self.outside.static_temp = outside_temperature + 273.15
        # Record and update current temperatures
        current_room_temperatures = {name: room.temperature for name, room in self.rooms.items()}

        # Set temperature sensors to record current temperature
        temperature_messages = self.read_temperature_sensors(current_time, noise_vector)

        # Send messages around and update controllers and regulation equipment
        controller_messages = self.update_control(temperature_messages, dt, current_time)

        outputs = self.update_heat_regulation(controller_messages, dt)

        self.update_room_temperatures(current_room_temperatures, outputs, dt)

        return (
            list(current_room_temperatures.values()),
            list(setpoints.values()),
            list(outputs.values()),
            temperature_messages,
            controller_messages,
        )

    def run_simulation(
            self,
            historical_data: pd.DataFrame,
            random_seed: Optional[int] = None,
    ) -> Tuple[npt.ArrayLike, List, List]:
        timeline = historical_data['timeline']
        temperature_data = historical_data['Lufttemperatur']
        dt = timeline[1] - timeline[0]
        simulation_length = min(len(timeline), len(temperature_data))

        self.room_simulation_startup(temperature_data[0])
        self._set_prng(random_seed)
        simulated_temperatures = np.zeros((simulation_length, len(self.rooms)))
        setpoints = np.zeros((simulation_length, len(self.controllers)))
        heat_reg_output = np.zeros((simulation_length, len(self.heat_reg_eq)))
        simulated_noise = self.generate_noise(simulated_temperatures.shape)

        accumulated_messages = {'temp': [], 'control': []}
        start = time.perf_counter()
        for i, (t, outside_temperature) in enumerate(zip(timeline, temperature_data)):
            (simulated_temperatures[i, :],
             setpoints[i, :],
             heat_reg_output[i, :],
             temperature_messages,
             controller_messages) = self.step(dt, t, outside_temperature, simulated_noise[i, :])
            accumulated_messages['temp'].append(temperature_messages)
            accumulated_messages['control'].append(controller_messages)
        stop = time.perf_counter()

        self.simulation_time = stop - start

        self.messages = accumulated_messages
        self.simulated_temperatures = simulated_temperatures
        self.simulated_setpoints = setpoints
        self.heat_reg_output = heat_reg_output
        self.timeline = timeline
        return simulated_temperatures, *self.messages # type: ignore


    def get_historical_data(self, filename: Union[str, Path]) -> pd.DataFrame:
        historical_data = pd.read_csv(filename, header=0, sep=',', parse_dates=[['Datum', 'Tid (UTC)']])
        df = historical_data[['Datum_Tid (UTC)', 'Lufttemperatur']].set_index('Datum_Tid (UTC)')
        df['timeline'] = np.array([3600 * float(i) for i, _ in enumerate(df.iterrows())])
        df = df.resample('10S').asfreq().interpolate('time')
        self.historical_data = df['timedelta'] = pd.to_timedelta(df['timeline'], unit='S')
        return df

    def plot_temperatures(self, temperature_data: pd.DataFrame):
        from matplotlib.cm import get_cmap
        from cycler import cycler
        color_list = get_cmap('tab10').colors
        # Formatting the x-axis correctly https://stackoverflow.com/questions/15240003/matplotlib-intelligent-axis-labels-for-timedelta
        plt.plot(temperature_data['timeline'], self.simulated_temperatures)
        plt.legend(self.rooms.keys())
        plt.figure()
        plt.rc('axes', prop_cycle=(cycler('color', color_list[2:])))
        plt.plot(temperature_data['timeline'], self.simulated_setpoints)
        plt.legend(self.controllers.keys())
        #plt.savefig(f'test_fig_seed_{self.random_seed}-2.png')
        plt.show()

    def save_simulation_data(self):
        simulation_data = pd.DataFrame()
        simulation_data['timeline'] = [rep_time for time in self.timeline for rep_time in itertools.repeat(time, 8)]
        simulation_data['datetime'] = [rep_date.isoformat() for date in self.historical_data.index
                                       for rep_date in itertools.repeat(date, 8)]
        simulation_data['messages_a'] = [json.dumps(message) for messages in self.messages[0]
                                         for message in messages
                                         if message[0]["n"].endswith('temp_sensor')]
        simulation_data['messages_b'] = [json.dumps(message) for messages in self.messages[1]
                                         for message in messages
                                         if message[0]["bn"].endswith('temp_sensor')]
        simulation_data['room_name'] =  [name for name, _ in zip(itertools.cycle(self.rooms.keys()), simulation_data['timeline'])]
        simulation_data['room_temperature'] = self.simulated_temperatures.flatten()
        simulation_data['setpoint'] = [item for row in self.simulated_setpoints
                                       for item in itertools.chain((float('nan'), float('nan')), row)]
        simulation_data['actuation'] = [item for row in self.heat_reg_output
                                        for item in itertools.chain((float('nan'), float('nan')), row)]
        simulation_data['previous_actuation'] = [float('nan'), float('nan'), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] \
                                                + [item for row in self.heat_reg_output[:-1]
                                                   for item in itertools.chain((float('nan'), float('nan')), row)]

        train_test_cutoff_index = len(simulation_data) * 5 // 6
        simulation_data.iloc[:train_test_cutoff_index, :].to_csv('simulation_data_train.csv', sep=';', index=False)
        simulation_data.iloc[train_test_cutoff_index:, :].to_csv('simulation_data_test.csv', sep=';', index=False)
        simulation_data.to_csv('simulation_data_all.csv', sep=';', index=False)

        return simulation_data

    def _set_prng(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            self.prng = RandomState(seed)
            random.seed(seed)
        else:
            self.prng = RandomState(np.random.randint(0, 2**32 - 1))

    def generate_noise(self, shape: Tuple[int, int]) -> npt.ArrayLike:
        return np.apply_along_axis(
                lambda m: np.convolve(m, np.ones((100, )) / 100, mode='same'),
                axis=0,
                arr=self.prng.randn(*shape),
        )

if __name__ == '__main__':
    pass
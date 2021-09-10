import csv
import sys
import datetime
from pathlib import Path
from typing import NamedTuple, Dict, Tuple, List, Sequence, Union, Optional, Generator, Iterable
import itertools
import random
import time

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
from scipy import signal
import pandas as pd
from matplotlib import pyplot as plt
import ujson as json

from equipment import (
    TemperatureSensor,
    HeatRegulationEquipment,
    Controller,
)
import a_sys_devices as Asys
import b_tech_devices as Btech
from room import Room
from utils import celsius


class TempSensorPair(NamedTuple):
    a_sys: TemperatureSensor
    b_tech: TemperatureSensor

    def temperature_service(self, current_time: float) -> "MessagePair":
        return MessagePair(
                self.a_sys.temperature_service(current_time),
                self.b_tech.temperature_service(current_time),
        )


class MessagePair(NamedTuple):
    a_sys: List[Dict]
    b_tech: List[Dict]


class ControllerPair(NamedTuple):
    a_sys: Controller
    b_tech: Controller

    def read_and_update(
            self,
            temperature_messages: MessagePair,
            dt: float,
    ):
        self.a_sys.read_and_update(temperature_messages.a_sys, dt)
        self.b_tech.read_and_update(temperature_messages.b_tech, dt)

    def regulation_messages(
            self,
            current_time: float,
    ) -> MessagePair:
        return MessagePair(
                self.a_sys.heater_message(current_time),
                self.b_tech.cooler_message(current_time),
        )

    def set_point(self, new_setpoint):
        self.a_sys.setpoint = new_setpoint
        self.b_tech.setpoint = new_setpoint - 273.15

    def get_setpoint(self):
        return self.a_sys.setpoint, self.b_tech.setpoint

class HeatRegPair(NamedTuple):
    a_sys: HeatRegulationEquipment
    b_tech: HeatRegulationEquipment

    def regulate_output(self, reg_messages: MessagePair) -> None:
        self.a_sys.regulate_output(reg_messages.a_sys)
        self.b_tech.regulate_output(reg_messages.b_tech)

    def produce(self) -> float:
        return self.a_sys.produce() + self.b_tech.produce()

    def output(self) -> Tuple[float, float]:
        return (
            self.a_sys.output,
            self.b_tech.output,
        )

class Environment:
    temp_sensors: Dict[str, TempSensorPair]
    controllers: Dict[str, ControllerPair]
    heat_reg_eq: Dict[str, HeatRegPair]
    rooms: Dict[str, Room]
    room_names = ('OO', 'CC', 'NW', 'NN', 'NE', 'SW', 'SS', 'SE')
    room_positions = [(2, 0), (0, 0), (1, -1), (1, 0), (1, 1), (-1, -1), (-1, 0), (-1, 1)]
    set_points: Dict[str, float]
    office_names = room_names[2:]
    office_positions = room_positions[2:]

    def __init__(
            self,
    ):
        self.room_init()
        self.equipment_init()

    def room_init(self):
        """ Setup all the rooms, returns a list containing all rooms [corridor, outside, *offices]

        The corridor layout is:
                    Outside/OO
          ----------------------------
         R2 | R0/NW | R1/NN | R2/NE | R0
          ----------------------------
                    Corridor/CC
          ----------------------------
         R5 | R3/SW | R4/SS | R5/SE | R3
          ----------------------------
                    Outside/OO
        """

        self.corridor = Room(
                name='CC',
                temperature=273.0,
                heat_capacity=46e3,
                position=(0, 0),
        )
        self.outside = Room(
                name='OO',
                temperature=273.0,
                heat_capacity=float('inf'),
                position=(2, 0),
                static=True
        )

        # Setup the offices
        self.offices = [
            Room(
                    name=name,
                    temperature=273.0,
                    heat_capacity=46e3,
                    position=pos,
            ) for name, pos in zip(self.office_names, self.office_positions)
        ]

        for office in (upper_offices := self.offices[:3]):
            office.add_neighbors(set(upper_offices) - {office})
        for office in (lower_offices := self.offices[3:]):
            office.add_neighbors(set(lower_offices) - {office})
        for office in self.offices:
            office.add_neighbors({self.corridor, self.outside})
        self.corridor.add_neighbors(set(self.offices))
        self.rooms = {room.name: room for room in itertools.chain((self.outside, self.corridor), self.offices)}

    def equipment_init(self):
        self.temp_sensors = {
            room.name: TempSensorPair(
                    Asys.TemperatureSensor(
                            f'{room.name}_temp_sensor',
                            room.temperature,
                    ),
                    Btech.TemperatureSensor(
                            'temp_sensor',
                            room.temperature,
                            coordinates=room.position,
                    )
            ) for room in self.rooms.values()
        }
        self.heat_reg_eq = {
            room.name: HeatRegPair(
                    Asys.Heater(
                            name=f'{room.name}_heater',
                            max_power=1500,
                    ),
                    Btech.Cooler(
                            name='cooler',
                            power=1500,
                            coordinates=room.position,
                    )
            ) for room in self.offices
        }
        self.controllers = {
            room.name: ControllerPair(
                    Asys.Controller(
                            setpoint=0,
                            k_I=150, k_P=0.5,
                            heater_id=f'{room.name}_heater',
                            indoor_temp_sensor_id=f'{room.name}_temp_sensor',
                    ),
                    Btech.Controller(
                            setpoint=0,
                            k_I=1.5, k_P=0.010,
                            cooler_id='cooler',
                            indoor_temp_sensor_id='temp_sensor',
                            coordinates=room.position,
                    )
            ) for room in self.offices
        }

    def step(
            self,
            dt: float,
            current_time: float,
            outside_temperature: float,
            noise_vector: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, List[Tuple[MessagePair, ...]]]:
        self.update_random_setpoints(current_time)
        self.outside.static_temp = outside_temperature + 273.15
        # Record and update current temperatures
        current_room_temperatures = {name: room.temperature for name, room in self.rooms.items()}

        # Set temperature sensors to record current temperature
        self.read_temperature_sensors(noise_vector)

        # Send messages around and update controllers and regulation equipment
        messages = self.update_control(dt, current_time)

        self.update_room_temperatures(current_room_temperatures, dt)

        return (
            list(current_room_temperatures.values()),
            list(controller.a_sys.setpoint for controller in self.controllers.values()),
            list(heat_reg.b_tech.output + heat_reg.a_sys.output for heat_reg in self.heat_reg_eq.values()),
            *messages
        )

    def _update_control(
            self,
            office: Room,
            dt: float,
            current_time: float
    ) -> Tuple[MessagePair, MessagePair]:
        temp_messages = self.temp_sensors[office.name].temperature_service(current_time)
        self.controllers[office.name].read_and_update(temp_messages, dt)
        reg_messages = self.controllers[office.name].regulation_messages(current_time)
        self.heat_reg_eq[office.name].regulate_output(reg_messages)

        return temp_messages, reg_messages

    def update_control(self, dt: float, current_time: float) -> List[Tuple[MessagePair, ...]]:
        return list(zip(*itertools.chain(
                (
                    msg_pair for msg_pair in (
                    self.temp_sensors['OO'].temperature_service(current_time),
                    self.temp_sensors['CC'].temperature_service(current_time)
                )
                ),
                (
                    msg_pair for room in self.offices
                    # Update the controllers and get messages
                    for msg_pair in self._update_control(room, dt, current_time)
                ),
        )))

    def read_temperature_sensors(self, noise_vector: npt.ArrayLike) -> None:
        for (name, temp_sensor), noise in zip(self.temp_sensors.items(), noise_vector):
            temp_sensor.a_sys.temperature = self.rooms[name].temperature + noise
            temp_sensor.b_tech.temperature = self.rooms[name].temperature + noise

    def update_room_temperatures(
            self,
            current_room_temperatures: Dict[str, float],
            dt: float
    ) -> None:
        for room_name, room in self.rooms.items():
            if room_name == 'OO':
                continue
            elif room_name == 'CC':
                room.update_temperature(
                        current_room_temperatures,
                        dt,
                        1e-4,
                        0.0,
                )
            else:
                room.update_temperature(
                        current_room_temperatures,
                        dt,
                        1e-4,
                        self.heat_reg_eq[room_name].produce(),
                )

    def _set_prng(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            self.prng = RandomState(seed)
            random.seed(seed)
        else:
            self.prng = RandomState(np.random.randint(0, 2**32 - 1))

    def generate_noise(self, shape: Tuple[int, int]) -> npt.ArrayLike:
        return np.apply_along_axis(
                lambda m: np.convolve(m, np.array([1] * 100) / 100, mode='same'),
                axis=0,
                arr=self.prng.randn(*shape),
        )

    def update_random_setpoints(self, timestep: float):
        if timestep % 3600 == 0:
            for controller_pair in self.controllers.values():
                controller_pair.set_point(random.uniform(278.15, 313.15))

    def room_simulation_startup(self, start_temperature: float):
        for room in self.rooms.values():
            if room.static:
                room.static_temp = start_temperature + 273.15
            else:
                room.temperature = start_temperature + 273.15

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

        accumulated_messages = []
        start = time.perf_counter()
        for i, (t, outside_temperature) in enumerate(zip(timeline, temperature_data)):
            (simulated_temperatures[i, :],
             setpoints[i, :],
             heat_reg_output[i, :],
             *messages) = self.step(dt, t, outside_temperature, simulated_noise[i, :])
            accumulated_messages.append(messages)
        stop = time.perf_counter()

        self.simulation_time = stop - start

        self.messages = tuple(system_messages for system_messages in zip(*accumulated_messages))
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
        plt.legend(self.room_names)
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
        simulation_data['room_name'] =  [name for name, _ in zip(itertools.cycle(self.room_names), simulation_data['timeline'])]
        simulation_data['room_temperature'] = self.simulated_temperatures.flatten()
        simulation_data['setpoint'] = [item for row in self.simulated_setpoints
                                       for item in itertools.chain((float('nan'), float('nan')), row)]
        simulation_data['actuation'] = [item for row in self.heat_reg_output
                                        for item in itertools.chain((float('nan'), float('nan')), row)]
        simulation_data['previous_actuation'] = [float('nan'), float('nan'), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\
                                                + [item for row in self.heat_reg_output[:-1]
                                                 for item in itertools.chain((float('nan'), float('nan')), row)]

        train_test_cutoff_index = len(simulation_data) * 5 // 6
        simulation_data.iloc[:train_test_cutoff_index, :].to_csv('simulation_data_train.csv', sep=';', index=False)
        simulation_data.iloc[train_test_cutoff_index:, :].to_csv('simulation_data_test.csv', sep=';', index=False)
        simulation_data.to_csv('simulation_data_all.csv', sep=';', index=False)

        return simulation_data


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['lines.linewidth'] = 0.5

    env = Environment(

    )

    temperature_data = env.get_historical_data('../SMHI-data/smhi-july-23-29-2018.csv')

    simulated_temperatures, messages_a, messages_b = env.run_simulation(
            temperature_data,
            random_seed=1337,
    )

    print(f'Time elapsed: {env.simulation_time:.2f} s.')

    simulation_data = env.save_simulation_data()

    #env.plot_temperatures(temperature_data)



import heapq
import random
from typing import Sequence, Dict, Tuple, Mapping, List, Union
from numbers import Real

from environment import Environment, MessageGroup, SensorGroup, Messages
from room import Room
import a_sys_devices as Asys
import b_tech_devices as Btech

class SingleRoomEnvironment(Environment):
    room_names = ('OO', 'AA')
    room_coordinates = ((0, 1), (0,0))
    setpoint_update_queue: List[Tuple[Real, str]] = [(0.0, name) for name in room_names[1:]] + [(float('inf'), 'OO')]


    def room_init(self):
        self.outside = Room(
                name=self.room_names[0],
                temperature=273.0,
                heat_capacity=float('inf'),
                position=self.room_coordinates[0],
                static=True
        )

        self.rooms = {'OO': self.outside} | {
            name: Room(
                    name=name,
                    temperature=273.0,
                    heat_capacity=46e3,
                    position=pos,
            ) for name, pos in zip(self.room_names[1:], self.room_coordinates[1:])
        }

        self.rooms['AA'].add_neighbors({(4, self.outside)})


    def equipment_init(self):
        self.temp_sensors = {
            room.name: {
                'heater': Asys.TemperatureSensor(
                        f'{room.name}_temp_sensor',
                        room.temperature,
                ),
                'cooler': Btech.TemperatureSensor(
                        'temp_sensor',
                        room.temperature,
                        coordinates=room.position,
                )
            } for room in self.rooms.values()
        }

        self.heat_reg_eq = {
            room.name: {
                'heater': Asys.Heater(
                        name=f'{room.name}_heater',
                        max_power=3000,
                ),
                'cooler': Btech.Cooler(
                        name='cooler',
                        power=3000,
                        coordinates=room.position,
                ),
            } for room in self.rooms.values()
            if room.name != self.room_names[0]
        }

        self.controllers = {
            room.name: {
                'heater': Asys.Controller(
                        setpoint=0,
                        k_I=1, k_P=100,
                        heater_id=f'{room.name}_heater',
                        indoor_temp_sensor_id=f'{room.name}_temp_sensor',
                ),
                'cooler': Btech.Controller(
                        setpoint=0,
                        k_I=0.05, k_P=1,
                        cooler_id='cooler',
                        indoor_temp_sensor_id='temp_sensor',
                        coordinates=room.position,
                ),
            } for room in self.rooms.values()
            if room.name != self.room_names[0]
        }


    def update_random_setpoints(self, current_time: float) -> Dict[str, float]:
        while (True):
            update_time, room_name = heapq.heappop(self.setpoint_update_queue)
            if update_time <= current_time:
                new_setpoint = random.uniform(278.15, 313.15)
                self.controllers[room_name]['heater'].update_setpoint(new_setpoint)
                self.controllers[room_name]['cooler'].update_setpoint(new_setpoint - 273.15)
                new_update_time = current_time + self.prng.poisson(3 * 3600)
                heapq.heappush(self.setpoint_update_queue, (new_update_time, room_name))
            else:
                heapq.heappush(self.setpoint_update_queue, (update_time, room_name))
                break
        return {
            room_name: controllers['heater'].setpoint
            for room_name, controllers in self.controllers.items()
        }


class EightRoomEnvironment(Environment):
    room_names = ('OO', 'A11', 'A12', 'A13', 'A14', 'A21', 'A22', 'A23', 'A24')
    room_coordinates = ((0, 2), (0, 0), (1, 0), (2, 0), (3, 0), (1, 0), (2, 0), (3, 0), (4, 0))
    setpoint_update_queue: List[Tuple[int, str]] = [(0, name) for name in room_names[1:]]
    """
       --------------------------------
       |A11|A12|A13|A14|
       ----------------- OO
       |A21|A22|A23|A24|
       -----------------
    """

    def room_init(self):
        self.outside = Room(
                name=self.room_names[0],
                temperature=273.0,
                heat_capacity=float('inf'),
                position=self.room_coordinates[0],
                static=True
        )

        self.rooms = {'OO': self.outside} | {
            name: Room(
                    name=name,
                    temperature=273.0,
                    heat_capacity=46e3,
                    position=pos,
            ) for name, pos in zip(self.room_names[1:], self.room_coordinates[1:])
        }

        # Bottom row
        self.rooms['A11'].add_neighbors({(2, self.outside), (1, self.rooms['A12']), (1, self.rooms['A21'])})
        self.rooms['A12'].add_neighbors(
                {(1, self.outside), (1, self.rooms['A11']), (1, self.rooms['A13']), (1, self.rooms['A22'])})
        self.rooms['A13'].add_neighbors(
                {(1, self.outside), (1, self.rooms['A12']), (1, self.rooms['A14']), (1, self.rooms['A23'])})
        self.rooms['A14'].add_neighbors({(2, self.outside), (1, self.rooms['A13']), (1, self.rooms['A24'])})
        # Top row
        self.rooms['A21'].add_neighbors({(2, self.outside), (1, self.rooms['A22']), (1, self.rooms['A11'])})
        self.rooms['A22'].add_neighbors(
                {(1, self.outside), (1, self.rooms['A21']), (1, self.rooms['A23']), (1, self.rooms['A12'])})
        self.rooms['A23'].add_neighbors(
                {(1, self.outside), (1, self.rooms['A22']), (1, self.rooms['A24']), (1, self.rooms['A13'])})
        self.rooms['A24'].add_neighbors({(2, self.outside), (1, self.rooms['A23']), (1, self.rooms['A14'])})

    def equipment_init(self):
        self.temp_sensors = {
            room.name: {
                    'heater': Asys.TemperatureSensor(
                            f'{room.name}_temp_sensor',
                            room.temperature,
                    ),
                    'cooler': Btech.TemperatureSensor(
                            'temp_sensor',
                            room.temperature,
                            coordinates=room.position,
                    )
            } for room in self.rooms.values()
        }

        self.heat_reg_eq = {
            room.name: {
                'heater': Asys.Heater(
                            name=f'{room.name}_heater',
                            max_power=1500,
                ),
                'cooler': Btech.Cooler(
                            name='cooler',
                            power=1500,
                            coordinates=room.position,
                ),
            } for room in self.rooms.values()
            if room.name != self.room_names[0]
        }

        self.controllers = {
            room.name: {
                'heater': Asys.Controller(
                        setpoint=0,
                        k_I=1, k_P=100,
                        heater_id=f'{room.name}_heater',
                        indoor_temp_sensor_id=f'{room.name}_temp_sensor',
                ),
                'cooler': Btech.Controller(
                        setpoint=0,
                        k_I=0.005, k_P=0.2,
                        cooler_id='cooler',
                        indoor_temp_sensor_id='temp_sensor',
                        coordinates=room.position,
                ),
            } for room in self.rooms.values()
            if room.name != self.outside.name
        }

    def update_random_setpoints(self, current_time: float) -> Dict[str, float]:
        while(True):
            update_time, room_name = heapq.heappop(self.setpoint_update_queue)
            if update_time <= current_time:
                new_setpoint = random.uniform(280.15, 310.15)
                self.controllers[room_name]['heater'].update_setpoint(new_setpoint)
                self.controllers[room_name]['cooler'].update_setpoint(new_setpoint - 273.15)
                new_update_time = current_time + self.prng.poisson(3600)
                heapq.heappush(self.setpoint_update_queue, (new_update_time, room_name))
            else:
                heapq.heappush(self.setpoint_update_queue, (update_time, room_name))
                break
        return {self.outside.name: float('nan')} | {
            room_name: controllers['heater'].setpoint
            for room_name, controllers in self.controllers.items()
        }





if __name__ == '__main__':
    import graphviz
    env = EightRoomEnvironment()
    #print(list(a for _, a in env.rooms['A11'].neighbors.values()))

    temperature_data = env.get_historical_data('../SMHI-data/smhi-july-23-29-2018.csv')

    simulated_temperatures, messages = env.run_simulation(
            temperature_data,
            random_seed=1337,
    )

    simulation_data = env.save_simulation_data()
    #env.plot_temperatures(temperature_data)
    """
    dot = graphviz.Digraph(comment='rooms')
    dot.node('OO')
    for name in env.rooms:
        dot.node(name)
    for name, room in env.rooms.items():
        for _, neighbor in room.neighbors.values():
            dot.edge(name, neighbor.name)
    print(dot.source)
    dot.render('room_graph.gv', view=True)
    """
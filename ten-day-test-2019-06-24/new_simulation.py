import heapq
import random
from typing import Sequence, Dict, Tuple, Mapping, List

from environment import Environment, MessageGroup, SensorGroup, Messages
from room import Room
import a_sys_devices as Asys
import b_tech_devices as Btech


class EightRoomEnvironment(Environment):
    room_names = ('A11', 'A12', 'A13', 'A14', 'A21', 'A22', 'A23', 'A24')
    room_coordinates = ((0, 0), (1, 0), (2, 0), (3, 0), (1, 0), (2, 0), (3, 0), (4, 0))
    setpoint_update_queue: List[Tuple[int, str]] = [(0, name) for name in room_names]

    def room_init(self):
        self.outside = Room(
                name='OO',
                temperature=273.0,
                heat_capacity=float('inf'),
                position=(0, 2),
                static=True
        )

        self.rooms = {
            name: Room(
                    name=name,
                    temperature=273.0,
                    heat_capacity=46e3,
                    position=pos,
            ) for name, pos in zip(self.room_names, self.room_coordinates)
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
        }

        self.controllers = {
            room.name: {
                'heater': Asys.Controller(
                        setpoint=0,
                        k_I=150, k_P=0.5,
                        heater_id=f'{room.name}_heater',
                        indoor_temp_sensor_id=f'{room.name}_temp_sensor',
                ),
                'cooler': Btech.Controller(
                        setpoint=0,
                        k_I=1.5, k_P=0.010,
                        cooler_id='cooler',
                        indoor_temp_sensor_id='temp_sensor',
                        coordinates=room.position,
                ),
            } for room in self.rooms.values()
        }

    def update_random_setpoints(self, current_time: float) -> Dict[str, float]:
        while(True):
            update_time, room_name = heapq.heappop(self.setpoint_update_queue)
            if update_time == current_time:
                new_setpoint = random.uniform(278.15, 313.15)
                self.controllers[room_name]['heater'].update_setpoint(new_setpoint)
                self.controllers[room_name]['cooler'].update_setpoint(new_setpoint)
                new_update_time = current_time + self.prng.poisson(3600)
                heapq.heappush(self.setpoint_update_queue, (new_update_time, room_name))
            else:
                heapq.heappush(self.setpoint_update_queue, (update_time, room_name))
                break
        return {
            room_name: controllers['heater'].setpoint
            for room_name, controllers in self.controllers.items()
        }





if __name__ == '__main__':
    import graphviz
    env = EightRoomEnvironment()
    #print(list(a for _, a in env.rooms['A11'].neighbors.values()))

    temperature_data = env.get_historical_data('../SMHI-data/smhi-july-23-29-2018.csv')

    simulated_temperatures, messages_a, messages_b = env.run_simulation(
            temperature_data,
            random_seed=1337,
    )
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
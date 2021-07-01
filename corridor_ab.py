from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import b_tech_devices as Btech
import a_sys_devices as Asys
from room_double_controller import Room, Office
from utils import celsius
import argparse
import json

class Environment():
    def __init__(self, timeline):
        self.room_init()

    def room_init(self):
        """ Setup all the rooms, returns a list containing all rooms [corridor, outside, *offices] """
        """ The corridor layout is:
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
        # Setup the 'normal' rooms
        corridor = Room('CC', 273, heat_capacity=46e3, 
                heater_temperature_sensor=Asys.TemperatureSensor('CC_temp_sensor', 0, coordinates = (0, 0)),
                cooler_temperature_sensor=Btech.TemperatureSensor('temp_sensor', 0, coordinates=(0, 0)))
        outside = Room('OO', None, heat_capacity=46e3, 
                heater_temperature_sensor=Asys.TemperatureSensor('OO_temp_sensor', 0, coordinates=(2, 0)),
                cooler_temperature_sensor=Btech.TemperatureSensor('temp_sensor', 0, coordinates=(2, 0)), static=True)

        # Setup the offices
        offices = []
        office_names = ['NW', 'NN', 'NE', 'SW', 'SS', 'SE']
        office_positions = [(1,-1), (1,0), (1,1), (-1,-1), (-1,0), (-1,1)]
        # I want another function to do this 
        # office_temperatures = [303, 300, 294.0, 297, 291, 288]
        # I want another function to do this 
        # office_requested_temperatures = [303, 300, 294.0, 297, 291, 288]#[293, 297, 295, 294, 298, 291]
        #print(office_requested_temperatures)

        # Create the new offices, currently with no temperature
        for name, position in zip(office_names, office_positions):
            heater_temp = Asys.TemperatureSensor(f'{name}_temp_sensor', 0, coordinates=position)
            heater = Asys.Heater(f'{name}_Heater', 1500, coordinates=position)
            heater_controller = Asys.Controller(f'{name}_Controller', 150, 1, 
                    heater.name, None, heater_temp.name, coordinates=position)
            cooler_temp = Btech.TemperatureSensor(f'temp_sensor', 0, coordinates=position)
            cooler = Btech.Cooler(f'Cooler', 1500, coordinates=position)
            cooler_controller = Btech.Controller(f'Controller', 1.5, 0.01, 
                    None, cooler.name, cooler_temp.name, coordinates=position)
            offices.append(Office(name, temperature=None, heat_capacity=46e3, position=position,
                cooler_temperature_sensor=cooler_temp, heater_temperature_sensor=heater_temp, 
                heater=heater, cooler=cooler, 
                cooler_controller=cooler_controller, heater_controller=heater_controller))
        num_offices = len(offices)

        # Neighbor the North and South sides
        for i, office in enumerate(offices):
            if not i % 2 == 0:
                pass
            if i >= num_offices // 2:
                offset = num_offices // 2
            else:
                offset = 0
            neighbor_indices = offset + ((i-1) % (num_offices // 2)), offset + (((i+1) % (num_offices // 2)))
            office.add_neighbors([offices[neighbor_index] for neighbor_index in neighbor_indices])
        # Neighbor the north and south sides to the corridor and outside
        for office in offices:
            office.add_neighbors([corridor, outside])
        
        #return [outside, corridor] + offices
        self.rooms = [outside, corridor] + offices

    def room_setup(self, start_temperatures, setpoints):
        for room, temp, setpoint in zip(self.rooms, start_temperatures, setpoints):
            print(room)
            room.change_temperature(temp)
            room.heater_temperature_sensor.measure_temperature(room)
            room.cooler_temperature_sensor.measure_temperature(room)
            if isinstance(room, Office):
                room.heater_controller.update_setpoint(celsius(setpoint))
                room.cooler_controller.update_setpoint(setpoint)

    def loop(self, dt, timestep):
        loop_a_messages = []
        loop_b_messages = []
        for room in self.rooms:
            if isinstance(room, Office):
                loop_a_messages.extend(room.update_control(dt, timestep)[0:2])
                loop_b_messages.extend(room.update_control(dt, timestep)[2:4])
            else:
                loop_a_messages.append(room.heater_temperature_sensor.temperature_service(timestep))
                loop_b_messages.append(room.cooler_temperature_sensor.temperature_service(timestep))
            room.update_temperature(dt, 1e-4)
        for room in self.rooms:
            room.execute_temperature()
        return np.array([room.temperature for room in self.rooms])[np.newaxis, :], loop_a_messages, loop_b_messages

    def run(self, timeline, temperature_data, profile):
        dt = timeline[1] - timeline[0]
        simulated_temperatures = np.zeros((min(len(timeline), len(temperature_data)), len(self.rooms)))
        message_a_list = []
        message_b_list = []
        for i, (t, temperature) in enumerate(zip(timeline, temperature_data)):
            if t % 86400 == 64800 or t == 0:
                for room in environment.rooms[2:]:
                    new_setpoint = profile[room.name]['nighttime setpoint']
                    room.heater_controller.update_setpoint(new_setpoint)
                    room.cooler_controller.update_setpoint(celsius(new_setpoint))
            elif t % 86400 == 21600:
                for room in environment.rooms[2:]:
                    new_setpoint = profile[room.name]['daytime setpoint']
                    room.heater_controller.update_setpoint(new_setpoint)
                    room.cooler_controller.update_setpoint(celsius(new_setpoint))
            self.rooms[0].temperature = temperature + 273.15
            simulated_temperatures[i, :], a_messages, b_messages = self.loop(dt, t)
            message_a_list.extend(a_messages)
            message_b_list.extend(b_messages)
        return simulated_temperatures, message_a_list, message_b_list

def get_historical_data(filename):
    historical_data = pd.read_csv(filename, header=0, sep=',', parse_dates=[['Datum', 'Tid (UTC)']])
    series = pd.Series(historical_data['Lufttemperatur'].to_numpy(), index=historical_data['Datum_Tid (UTC)']).resample('10S').asfreq().interpolate('time')
    return series

if __name__ == '__main__':
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestep', help='Simulation timestep', type=int, default=10)
    parser.add_argument('--simulation-length', help='length of simulation in days', type=float, default=10)
    parser.add_argument('--output-folder', help='Output folder', default='.')
    parser.add_argument('--temperature-data', help='choose what real data you want', default='feb')
    parser.add_argument('--noise', help='simulate with noise', type=bool, default=False)
    parser.add_argument('--profile', help='json-object containing the simulation profile')
    args = parser.parse_args()

    if args.temperature_data == 'feb':
        temperature_data = get_historical_data('../SMHI-data/smhi-february-05-12-2018.csv')
    elif args.temperature_data == 'jul':
        temperature_data = get_historical_data('../SMHI-data/smhi-july-23-29-2018.csv')
    else:
        raise ValueError('Temperature data has to be \'feb\' or \'jul\'')

    dt = args.timestep
    timestop = 24*3600*args.simulation_length
    timeline = np.arange(0, timestop, dt)

    environment = Environment(timeline)
    environment.rooms[0].temperature = 290
    environment.room_setup(np.array([290, 290, 290, 290, 290, 290, 290, 290]), np.array([297, 296, 295, 294, 293, 292, 291, 290]))
    with open(args.profile) as profile_file:
        simulation_profile = json.load(profile_file)
    for room in environment.rooms:
        print(room, room.neighbors)
    start = time.time()
    simulated_temperatures, message_a_list, message_b_list = environment.run(timeline, temperature_data, simulation_profile)
    end = time.time()
    print("Time taken: ", end - start)
    #pprint(message_list)
    print(len(message_a_list))
    #pprint(message_list[-20:])
    if True:
        with open(f'{args.output_folder}/ab_messages_a.msg', 'w') as f:
            f.writelines(f'{message}\n' for message in message_a_list)
        with open(f'{args.output_folder}/ab_messages_b.msg', 'w') as f:
            f.writelines(f'{message}\n' for message in message_b_list)
    print(temperature_data.index)
    plt.plot(temperature_data.index[:simulated_temperatures.shape[0]], simulated_temperatures)
    plt.legend(('OO', 'CC', 'NW', 'NN', 'NE', 'SW', 'SS', 'SE'))
    plt.show()

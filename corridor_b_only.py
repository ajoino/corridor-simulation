from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import time
import b_tech_devices as Btech
from room import Room, Office

def celsius(kelvin):
    return kelvin - 273.15
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
        corridor = Room('CC', 273, heat_capacity=46e3, temperature_sensor=Btech.TemperatureSensor('temp_sensor', 0, coordinates = (0, 0)))
        outside = Room('OO', None, heat_capacity=46e3, temperature_sensor=Btech.TemperatureSensor('OO_temp_sensor', 0, coordinates=(2, 0)), static=True)

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
            temp = Btech.TemperatureSensor(f'{name}_temp_sensor', 0, coordinates=position)
            heater = Btech.Heater(f'{name}_Heater', 1500, coordinates=position)
            cooler = Btech.Cooler(f'{name}_Cooler', 1500, coordinates=position)
            controller = Btech.Controller(f'{name}_Controller', 1.50, 0.01, 
                    heater.name, cooler.name, temp.name, coordinates=position)
            offices.append(Office(name, temperature=None, heat_capacity=46e3, position=position,
                temperature_sensor=temp, heater=heater, cooler=cooler, controller=controller))
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
            room.change_temperature(temp)
            room.temperature_sensor.measure_temperature(room)
            if isinstance(room, Office):
                room.controller.update_setpoint(setpoint)

    def loop(self, dt, timestep):
        loop_messages = []
        for room in self.rooms:
            if isinstance(room, Office):
                loop_messages += room.update_control(dt, timestep)
            else:
                loop_messages.append(room.temperature_sensor.temperature_service(timestep))
            room.update_temperature(dt, 1e-4)
        for room in self.rooms:
            room.execute_temperature()
        return np.array([room.temperature for room in self.rooms])[np.newaxis, :], loop_messages

    def run(self, timeline):
        dt = timeline[1] - timeline[0]
        simulated_temperatures = np.zeros((len(timeline), len(self.rooms)))
        message_list = []
        for i, t in enumerate(timeline):
            if t % 86400 == 43200:
                for room in environment.rooms[2:]:
                    room.controller.update_setpoint(290-273.15)
            elif t % 86400 == 0:
                for room in environment.rooms[2:]:
                    room.controller.update_setpoint(295-273.15)
            self.rooms[0].temperature = 293 + 10*np.sin(2*np.pi*t/timeline[-1])
            if i % 100 == 0:
                print(self.rooms[0].temperature)
            simulated_temperatures[i, :], messages = self.loop(dt, t)
            message_list.extend(messages)
        return simulated_temperatures, message_list

if __name__ == '__main__':
    np.random.seed(42)
    dt = 10
    timestop = 24*3600*1
    timeline = np.arange(0, timestop, dt)

    environment = Environment(timeline)
    environment.rooms[0].temperature = 290-273.15
    environment.room_setup(np.array([290, 291, 292, 293, 294, 295, 296, 297]), np.array([297, 296, 295, 294, 293, 292, 291, 290]))
    print([room.controller.setpoint for room in environment.rooms[2:]])
    
    start = time.time()
    simulated_temperatures, message_list = environment.run(timeline)
    end = time.time()
    print("Time taken: ", end - start)
    #pprint(message_list)
    print(len(message_list))
    #pprint(message_list[-20:])
    pprint(message_list[0])
    pprint(message_list[-20])
    if True:
        with open('b_only_messages.msg', 'w') as f:
            f.writelines(f'{message}\n' for message in message_list)
    plt.plot(timeline, simulated_temperatures)
    plt.legend(('OO', 'CC', 'NW', 'NN', 'NE', 'SW', 'SS', 'SE'))
    plt.show()

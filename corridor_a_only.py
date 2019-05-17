import numpy as np
import a_sys_devices as Asys
from room import Room, Office

class Environment():
    def __init__(self, timeline):
        #self.rooms = rooms
        self.timeline = timeline
        self.dt = timeline[1] - timeline[0]
        #self.simulated_temperatures = np.zeros((len(timeline), len(self.rooms)))
        self.rooms = self.room_init()

    def room_init(self):
        """ Setup all the rooms, returns a list containing all rooms [corridor, outside, *offices] """
        """ The corridor layout is:
                 Outside
          --------------------
         R2 | R0 | R1 | R2 | R0
          --------------------
                Corridor
          --------------------
         R5 | R3 | R4 | R5 | R3
          --------------------
                 Outside
        """
        # Setup the 'normal' rooms
        corridor = Room('CC', 273, temperature_sensor=Asys.TemperatureSensor('CC_temp_sensor', 0))
        outside = Room('OO', 298, temperature_sensor=Asys.TemperatureSensor('OO_temp_sensor', 0), static=True)

        # Setup the offices
        offices = []
        office_names = ['NW', 'NN', 'NE', 'SW', 'SS', 'SE']
        # I want another function to do this 
        # office_temperatures = [303, 300, 294.0, 297, 291, 288]
        # I want another function to do this 
        # office_requested_temperatures = [303, 300, 294.0, 297, 291, 288]#[293, 297, 295, 294, 298, 291]
        #print(office_requested_temperatures)

        # Create the new offices, currently with no temperature
        for name in office_names:
            temp = Asys.TemperatureSensor(f'{name}_temp_sensor', 0)
            heater = Asys.Heater(f'{name}_Heater', 1500)
            cooler = Asys.Cooler(f'{name}_Cooler', 1500)
            offices.append(Office(name, temperature=None, 
                temperature_sensor=temp, heater=heater, cooler=cooler))
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
        
        return [outside, corridor] + offices

    def room_setup(self, start_temperatures):
        for room, temp in zip(self.rooms, start_temperatures):
            room.temperature = temp
            room.temperature_sensor.measure_temperature(room)

if __name__ == '__main__':
    dt = 10
    timestop = 2000
    timeline = np.arange(0, timestop, dt)

    environment = Environment(timeline)
    environment.room_setup([293, 293, 293, 293, 293, 293, 293, 293])
    for room in environment.rooms:
        print(room.temperature_sensor.temperature_service())

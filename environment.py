from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from room import Room, Office
import random

class Environment():
    def __init__(self, rooms, timeline):
        self.rooms = rooms
        self.timeline = timeline
        self.dt = timeline[1] - timeline[0]
        self.simulated_temperatures = np.zeros((len(timeline), len(self.rooms)))

    def run_loop(self):
        for room in self.rooms:
            room.update_temperature(self.dt,0.0003)
        for room in self.rooms:
            room.execute_temperature()
        return np.array([room.temperature for room in self.rooms])

    def run_simulation(self):
        for room in rooms[2:]:
            room.heater.turn_on()
            room.cooler.turn_on()
        for i, t in enumerate(timeline):
            self.simulated_temperatures[i, :] = self.run_loop()
            rooms[0].temperature = 285.5 + 12.5*np.sin(2*np.pi*t/timeline[-1])
            self.simulated_temperatures[i, 0] = rooms[0].temperature
            for room in self.rooms[2:]:
                room.heater.regulate(room.temperature, room.requested_temperature)
                room.cooler.regulate(room.temperature, room.requested_temperature)
                """
                if room.temperature >= room.requested_temperature:
                    room.heater.turn_off()
                    if room.temperature > room.requested_temperature + 0.5:
                        room.cooler.turn_on()
                elif room.temperature < room.requested_temperature:
                    room.cooler.turn_off()
                    if room.temperature < room.requested_temperature - 0.5:
                        room.heater.turn_on()
                """

if __name__ == '__main__':
    offices = []
    corridor = Room('CC', 273)
    outside = Room('OO', 298, static=True)
    office_names = ['NW', 'NN', 'NE', 'SW', 'SS', 'SE']
    office_temperatures = [303, 300, 294.0, 297, 291, 288]
    office_requested_temperatures = [303, 300, 294.0, 297, 291, 288]#[293, 297, 295, 294, 298, 291]
    print(office_requested_temperatures)


    for name, temp, req in zip(office_names, office_temperatures, office_requested_temperatures):
        offices.append(Office(name, temperature=temp, requested_temperature=req, heater_power=1500/43.2e3, cooler_power=1500/43.2e3))
    num_offices = len(offices)

    for i, office in enumerate(offices):
        if not i % 2 == 0:
            pass
        if i >= num_offices // 2:
            offset = num_offices // 2
        else:
            offset = 0
        neighbor_indices = offset + ((i-1) % (num_offices // 2)), offset + (((i+1) % (num_offices // 2)))
        office.add_neighbors([offices[neighbor_index] for neighbor_index in neighbor_indices])

    for office in offices:
        office.add_neighbors([corridor, outside])

    dt = 1
    seconds_per_day = 24 * 3600
    timeline = np.arange(0, seconds_per_day, dt)
    #timeline = np.linspace(0,seconds_per_day,seconds_per_day//dt)

    rooms = [outside, corridor] + offices
    requested_temperatures = np.ones((len(timeline), len(offices))) * np.array(office_requested_temperatures)
    environment = Environment(rooms, timeline)
    environment.run_simulation()
    print([room.heater.total_power_used for room in environment.rooms[2:]])
    print([room.cooler.total_power_used for room in environment.rooms[2:]])
    plt.plot(timeline, environment.simulated_temperatures)
    plt.legend(('OO', 'CC', 'NW', 'NN', 'NE', 'SW', 'SS', 'SE'))
    plt.plot(timeline, requested_temperatures, 'k')
    plt.show()
    """
    corridor_temperature_over_time = [corridor.temperature]
    office_temperature_over_time = [offices[0].temperature]
    outside_temperature_over_time = [outside.temperature]
    for t in timeline[:-1:]:
        for office in offices:
            office.update_temperature(dt, 0.003)
        corridor.update_temperature(dt, 0.003)
        for office in offices:
            office.execute_temperature()
        corridor.execute_temperature()
        corridor_temperature_over_time.append(corridor.temperature)
        office_temperature_over_time.append(offices[0].temperature)
        outside_temperature_over_time.append(outside.temperature)
    plt.plot(timeline, corridor_temperature_over_time, timeline, office_temperature_over_time, timeline, outside_temperature_over_time)
    plt.show()
    """

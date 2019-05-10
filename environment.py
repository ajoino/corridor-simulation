from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from room import Room, Office

class Environment():
    def __init__(self):
        pass

if __name__ == '__main__':
    offices = []
    corridor = Room('CC', 273)
    outside = Room('OO', 298)
    office_names = ['NW', 'NN', 'NE', 'SW', 'SS', 'SE']

    for name in office_names:
        offices.append(Office(name, 295, heater_power=1500, cooler_power=200))
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
    timeline = np.arange(0,611,dt)
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

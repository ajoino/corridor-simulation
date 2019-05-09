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

    timeline = np.arange(0,201,10)
    print(timeline)
    corridor_temperature_over_time = [corridor.temperature]
    for t in timeline:
        temp = corridor.temperature
        neighbor_temps = [neighbor.temperature for neighbor in corridor.neighbors]
        corridor.temperature = temp + 10*0.008*np.sum([neighbor_temp - temp for neighbor_temp in neighbor_temps])
        corridor_temperature_over_time.append(corridor.temperature)
        if t == 0:
            print(neighbor_temps)
    print(corridor_temperature_over_time)

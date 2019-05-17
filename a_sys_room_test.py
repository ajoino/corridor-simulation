import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import cProfile, pstats, io
from pstats import SortKey
import a_sys_devices as Asys
from room import Room, Office

heater = Asys.Heater('Heater', 5000)
cooler = Asys.Cooler('Cooler', 1000)
temp = Asys.TemperatureSensor('Temp', temperature=293)
    #def __init__(self, setpoint, k_P, k_I, 
            #heater_id=None, cooler_id=None, indoor_temp_sensor_id='', outdoor_temp_sensor_id=''):
controller = Asys.Controller(298, 2000, 20.0, heater.name, cooler.name, temp.name)
    #def __init__(self, name='', temperature=None, heat_capacity=1, 
            #temperature_sensor=None, position=None, heater=None, cooler=None):
office = Office(name='Asys room', temperature=293, heat_capacity=46.3e3, heater=heater, cooler=cooler, temperature_sensor=temp)
room = Room('Outside', 290, static=True)
office.add_neighbors([room])

dt = 1
time_stop = 2000#24*3600
timeline = np.arange(0, time_stop + dt, dt)

pr = cProfile.Profile()

print(office.temperature_sensor)
office_temp = np.zeros((len(timeline), ))#[office.temperature]
office_temp[0] = office.temperature
profile_it = True
if profile_it:
    pr.enable()
start = int(time.time())
for i, t in enumerate(timeline[1:], 1):
    controller.read_and_update(office.temperature_sensor.temperature_service(), dt)
    office.update_temperature(dt, 0.01)
    office.execute_temperature()
    office.heater.regulate_output(controller.heater_message())
    office.cooler.regulate_output(controller.cooler_message())
    #print(controller.control, office.heater.produce(), office.cooler.produce(), office.temperature - 298)
    #if i % 360:
    #pprint(controller.heater_message())
    #office_temp.append(office.temperature)
    office_temp[i] = office.temperature
if profile_it:
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
stop = time.time()
print(stop - start)

plt.plot(timeline, office_temp)
plt.show()


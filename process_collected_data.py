import os
import json
import re
import numpy as np

eps = 1e-12


# ['collected_data', 'collected_data_autopilot', 'collected_data_customized']
data_dir = 'collected_data'


# weather_indexes = [11, 19]
# routes = [19, 29, 39, 49, 59, 69]
# weather_indexes = [0]
# routes = [11, 12, 13, 14, 15, 16]

weather_indexes = [15]
routes = [i for i in range(76)]

# infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'vehicle_blocked']
infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'off_road']
for weather_id in weather_indexes:
    for route in routes:
        print(route)
        route_str = str(route)
        if route < 10:
            route_str = '0'+route_str

        parent_folder = data_dir+'/'+'route_'+route_str+'_'+str(weather_id)
        events_path = parent_folder + '/' + 'events.txt'
        measurements_path = parent_folder + '/' + 'measurements.csv'
        measurements_loc_path = parent_folder + '/' + 'measurements_loc.csv'
        new_measurements_path = parent_folder + '/' + 'driving_log.csv'

        events_list = []

        with open(events_path) as json_file:
            events = json.load(json_file)

        infractions = events['_checkpoint']['records'][0]['infractions']
        for infraction_type in infraction_types:
            for infraction in infractions[infraction_type]:
                loc = re.search('.*x=(.*), y=(.*), z=(.*)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    events_list.append((x, y, infraction_type))

        num_of_lines = 0
        with open(measurements_path, 'r') as f_in:
            with open(measurements_loc_path, 'r') as f_loc_in:
                measurements = f_in.read().split('\n')
                locations = f_loc_in.read().split('\n')

        print('-'*100, route, weather_id, len(measurements), '-'*100)
        misbehavior_list = []
        with open(new_measurements_path, 'w') as f_out:
            for i in range(len(measurements)):
                new_line = ''
                if i == 0:
                    # TBD: include topdown in the title_row
                    title_row = ','.join(['FrameId', 'far_command', 'speed', 'steering', 'throttle', 'brake', 'center', 'left', 'right', 'x', 'y', 'Misbehavior', 'Crashed'])
                    new_line = title_row
                else:
                    m_i = measurements[i].split(',')
                    l_i = locations[i].split(',')
                    if m_i != [''] and l_i != ['']:

                        # we use crashed to represent all violations
                        crashed = 0
                        x, y = float(l_i[0]), float(l_i[1])
                        misbehavior_name = ''

                        for event in events_list:
                            x_e, y_e, event_name = event
                            if np.abs(x_e-x) < eps and np.abs(y_e-y) < eps:
                                misbehavior_name += '_'+event_name
                                crashed = 1
                                misbehavior_list.append((misbehavior_name, i))

                        data_row = m_i+l_i+[misbehavior_name, str(crashed)]
                        new_line = ','.join(data_row)
                f_out.write(new_line+'\n')
            # print misbehaviors and the corresponding frame ids
            vehicle_blocked_start = False
            for j, (misbehavior_name, i) in enumerate(misbehavior_list):
                if misbehavior_name != '_vehicle_blocked':
                    print(misbehavior_name, i)
                    vehicle_blocked_start = False
                else:
                    if not vehicle_blocked_start:
                        print(misbehavior_name + ' starts', i)
                        vehicle_blocked_start = True
                    if j == len(misbehavior_list)-1:
                        print(misbehavior_name + ' ends', i)


import sys
sys.path.append("../carla_0994_no_rss/PythonAPI/carla")
sys.path.append("scenario_runner")
import xml.etree.ElementTree as ET
import pathlib
from leaderboard.leaderboard.utils.route_parser import RouteParser
import os

def calibrate(yaw):
    if yaw < 0:
        yaw += 360
    elif yaw >= 360:
        yaw -= 360
    return int(yaw)


TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
%s
</routes>"""
scenario_file = 'leaderboard/data/customized_scenarios.json'
scenario_types = ['Scenario4']
town_names = ['Town01']


world_annotations = RouteParser.parse_annotations_file(scenario_file)
pitch = 0
roll = 0

for town_name in town_names:
    scenarios = world_annotations[town_name]
    for scenario in scenarios:
        if "scenario_type" not in scenario:
            continue
        scenario_name = scenario["scenario_type"]
        if scenario_name in scenario_types:
            route_ids = {'front':0, 'left':0, 'right':0}
            for event in scenario["available_event_configurations"]:
                trigger_waypoint = event['transform']
                RouteParser.convert_waypoint_float(trigger_waypoint)


                trigger_x = trigger_waypoint['x']
                trigger_y = trigger_waypoint['y']
                trigger_yaw = trigger_waypoint['yaw']
                trigger_z = trigger_waypoint['z']

                yaw_to_trigger_xy_offsets = {0:(-1, 0), 90:(0, -1), 180:(1, 0), 270:(0, 1)}
                yaw_to_actor_xy_offsets = {0:(0, 1), 90:(-1, 0), 180:(0, -1), 270:(1, 0)}
                yaw_to_post_actor_xy_offsets = {0:(1, 0), 90:(0, 1), 180:(-1, 0), 270:(0, -1)}
                direction_to_actor_yaw_offset = {'front':0, 'left':-90, 'right':90}

                trigger_dx, trigger_dy = yaw_to_trigger_xy_offsets[calibrate(trigger_yaw)]
                trigger_l = 10


                pre_trigger_x = trigger_x + trigger_dx*trigger_l
                pre_trigger_y = trigger_y + trigger_dy*trigger_l

                if 'other_actors' in event:
                    other_vehicles = event['other_actors']
                    directions = ['front', 'left', 'right']
                    for dir in directions:
                        if dir in other_vehicles:
                            route_id = route_ids[dir]
                            for actor_waypoint in other_vehicles[dir]:
                                RouteParser.convert_waypoint_float(actor_waypoint)

                                actor_yaw = calibrate(trigger_yaw + direction_to_actor_yaw_offset[dir])
                                actor_z = actor_waypoint['z']


                                actor_dx, actor_dy = yaw_to_actor_xy_offsets[calibrate(actor_yaw)]
                                post_actor_dx, post_actor_dy = yaw_to_post_actor_xy_offsets[calibrate(actor_yaw)]
                                actor_l = 4
                                post_actor_l = 10

                                actor_x = actor_waypoint['x'] + actor_dx*actor_l
                                actor_y = actor_waypoint['y'] + actor_dy*actor_l

                                post_actor_x = actor_x + post_actor_dx*post_actor_l
                                post_actor_y = actor_y + post_actor_dy*post_actor_l


                                start_str = '<route id="{}" town="{}">\n'.format(route_id, town_name)
                                waypoint_template = '    <waypoint pitch="{}" roll="{}" x="{}" y="{}" yaw="{}" z="{}" />\n'
                                end_str = '</route>'

                                w1 = waypoint_template.format(pitch, roll, pre_trigger_x, pre_trigger_y, trigger_yaw, trigger_z)
                                w2 = waypoint_template.format(pitch, roll, trigger_x, trigger_y, trigger_yaw, trigger_z)
                                w3 = waypoint_template.format(pitch, roll, actor_x, actor_y, actor_yaw, actor_z)
                                w4 = waypoint_template.format(pitch, roll, post_actor_x, post_actor_y, actor_yaw, actor_z)

                                waypoints_str = w1+w2+w3+w4

                                route_str = start_str+waypoints_str+end_str


                                parent_folder = 'leaderboard/data/customized_routes'
                                folder_1 = parent_folder+'/'+town_name
                                folder_2 = folder_1+'/'+scenario_name
                                folder_3 = folder_2+'/'+dir
                                if not os.path.exists(parent_folder):
                                    os.mkdir(parent_folder)
                                if not os.path.exists(folder_1):
                                    os.mkdir(folder_1)
                                if not os.path.exists(folder_2):
                                    os.mkdir(folder_2)
                                if not os.path.exists(folder_3):
                                    os.mkdir(folder_3)

                                route_id_str = str(route_id)
                                if route_id < 10:
                                    route_id_str = '0'+route_id_str
                                pathlib.Path(folder_3+'/route_{}.xml'.format(route_id_str)).write_text(TEMPLATE % route_str)

                                route_ids[dir] += 1




                else:
                    # TBD
                    other_vehicles = None

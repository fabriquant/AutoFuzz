import os
from ga_fuzzing import run_simulation
from object_types import pedestrian_types, vehicle_types, static_types, vehicle_colors
import random
import pickle
import numpy as np
from datetime import datetime
from customized_utils import make_hierarchical_dir, convert_x_to_customized_data, exit_handler, customized_routes, parse_route_and_scenario
import atexit



os.environ['HAS_DISPLAY'] = '0'
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
port = 2033
save_rerun_cur_info = False

def rerun_simulation(pickle_filename, is_save, rerun_save_folder, ind, scenario_file, ego_car_model='lbc'):
    cur_folder = rerun_save_folder+'/'+str(ind)
    if not os.path.exists(cur_folder):
        os.mkdir(cur_folder)


    is_bug = False

    # parameters preparation
    if ind == 0:
        launch_server = True
    else:
        launch_server = False

    with open(pickle_filename, 'rb') as f_in:
        d = pickle.load(f_in)['info']
        x = d['x']
        x[-1] = port
        waypoints_num_limit = d['waypoints_num_limit']
        max_num_of_static = d['num_of_static_max']
        max_num_of_pedestrians = d['num_of_pedestrians_max']
        max_num_of_vehicles = d['num_of_vehicles_max']
        customized_center_transforms = d['customized_center_transforms']

        if 'parameters_min_bounds' in d:
            parameters_min_bounds = d['parameters_min_bounds']
            parameters_max_bounds = d['parameters_max_bounds']
        else:
            parameters_min_bounds = None
            parameters_max_bounds = None


        episode_max_time = 50
        call_from_dt = d['call_from_dt']
        town_name = d['town_name']
        scenario = d['scenario']
        direction = d['direction']
        route_str = d['route_str']




    folder = '_'.join([town_name, scenario, direction, route_str])

    rerun_folder = make_hierarchical_dir(['rerun', folder])

    # extra
    # hack for now before we saved this field
    route_type = 'town05_right_0'
    # customized_d = customized_bounds_and_distributions[scenario_type]
    route_info = customized_routes[route_type]

    location_list = route_info['location_list']

    parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file)




    customized_data = convert_x_to_customized_data(x, waypoints_num_limit, max_num_of_static, max_num_of_pedestrians, max_num_of_vehicles, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds)
    objectives, loc, object_type, info, save_path = run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model, rerun=True, rerun_folder=cur_folder)


    if objectives[0] > 0.2 or objectives[4] or objectives[5]:
        is_bug = True


    cur_info = {'x':x, 'objectives':objectives, 'loc':loc, 'object_type':object_type}

    if save_rerun_cur_info:
        with open(os.path.join(cur_folder, str(ind)), 'wb') as f_out:
            pickle.dump(cur_info, f_out)

    # copy data to another place
    if is_save:
        try:
            shutil.copytree(save_path, cur_folder)
        except:
            print('fail to copy from', save_path)

    return is_bug, objectives


if __name__ == '__main__':


    random.seed(0)
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    # ['bugs', 'non_bugs']
    data = 'bugs'
    # ['train', 'test', 'all']
    mode = 'test'


    rerun_save_folder = make_hierarchical_dir(['rerun', data, mode, time_str])

    scenario_folder = 'scenario_files'
    if not os.path.exists('scenario_files'):
        os.mkdir(scenario_folder)
    scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

    atexit.register(exit_handler, [port], rerun_save_folder, scenario_file)



    # if data == 'bugs':
    #     folder = 'bugs/False/nsga2/Town03/Scenario12/front/00/2020_08_11_20_07_07_used_for_retraining'
    # elif data == 'non_bugs':
    #     folder = 'non_bugs/False/nsga2/Town03/Scenario12/front/00/2020_08_11_20_07_07_used_for_retraining'
    #
    # subfolder_names = [sub_folder_name for sub_folder_name in os.listdir(folder)]
    # random.shuffle(subfolder_names)
    #
    #
    # assert len(subfolder_names) >= 2
    # mid = int(len(subfolder_names)//2)
    #
    # train_subfolder_names = subfolder_names[:mid]
    # test_subfolder_names = subfolder_names[mid:]
    #
    #
    #
    # if mode == 'train':
    #     chosen_subfolder_names = train_subfolder_names
    # elif mode == 'test':
    #     chosen_subfolder_names = test_subfolder_names
    # elif mode == 'all':
    #     chosen_subfolder_names = subfolder_names
    #
    # bug_num = 0
    # objectives_avg = None
    #
    # for ind, sub_folder_name in enumerate(chosen_subfolder_names):
    #     print('episode:', ind, '/', len(chosen_subfolder_names))
    #     sub_folder = os.path.join(folder, sub_folder_name)
    #     if os.path.isdir(sub_folder):
    #         for filename in os.listdir(sub_folder):
    #             if filename.endswith(".pickle"):
    #                 pickle_filename = os.path.join(sub_folder, filename)
    #                 is_bug, objectives = rerun_simulation(pickle_filename, True, rerun_save_folder, ind)
    #
    #                 if ind == 0:
    #                     objectives_avg = np.array(objectives)
    #                 else:
    #                     objectives_avg += np.array(objectives)
    #
    #                 if is_bug:
    #                     bug_num += 1
    #
    # print('bug_ratio :', bug_num / len(chosen_subfolder_names))
    # print('objectives_avg :', objectives_avg / len(chosen_subfolder_names))



    # pickle_filenames = ['data_for_analysis/other_controllers/2020_08_26_11_39_16_autopilot_pid1/bugs/35/cur_info.pickle']
    pickle_filenames = ['data_for_analysis/other_controllers/2020_08_26_11_39_22_pid_pid2/bugs/39/cur_info.pickle']

    # pickle_filename = 'data_for_analysis/new_nsga2-un_new_town05_right_50_15/2020_08_26_18_43_03_nsga2-un/bugs/3/cur_info.pickle'

    # pickle_filename =

    # pickle_filenames = ['data_for_analysis/high_dim_scene/2020_08_27_11_47_58_nsga2-un/bugs/'+str(ind)+'/cur_info.pickle' for ind in [10, 20, 33, 43]]

    for i, pickle_filename in enumerate(pickle_filenames):
        is_bug, objectives = rerun_simulation(pickle_filename, True, rerun_save_folder, i, scenario_file, ego_car_model='pid_agent')

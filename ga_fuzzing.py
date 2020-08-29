# hack: increase the maximum number of files to open to avoid too many files open error due to leakage.
# import resource
#
# print("getrlimit before:", resource.getrlimit(resource.RLIMIT_NOFILE))
# resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
# print("getrlimit:", resource.getrlimit(resource.RLIMIT_NOFILE))


import sys
import os
sys.path.append('pymoo')
carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')
sys.path.append('.')
sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
sys.path.append('scenario_runner')
sys.path.append('carla_project/src')




from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination, NoDuplicateElimination

from pymoo.model.population import Population
from pymoo.model.evaluator import Evaluator

from pymoo.algorithms.nsga2 import NSGA2, binary_tournament
from pymoo.algorithms.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.algorithms.random import RandomAlgorithm

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover

from pymoo.performance_indicator.hv import Hypervolume

import matplotlib.pyplot as plt

from object_types import WEATHERS, pedestrian_types, vehicle_types, static_types, vehicle_colors, car_types, motorcycle_types, cyclist_types

from customized_utils import create_transform, rand_real,  convert_x_to_customized_data, make_hierarchical_dir, exit_handler, arguments_info, is_critical_region, setup_bounds_mask_labels_distributions_stage1, setup_bounds_mask_labels_distributions_stage2, customize_parameters, customized_bounds_and_distributions, static_general_labels, pedestrian_general_labels, vehicle_general_labels, waypoint_labels, waypoints_num_limit, if_violate_constraints, customized_routes, parse_route_and_scenario, get_distinct_data_points, is_similar, check_bug, is_distinct, filter_critical_regions


from collections import deque


import numpy as np
import carla

from leaderboard.fuzzing import LeaderboardEvaluator
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.customized.object_params import Static, Pedestrian, Vehicle

import traceback
import json
import re
import time
from datetime import datetime

import pathlib
import shutil
import dill as pickle
# import pickle
import argparse
import atexit
import traceback
import math



import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.model.individual import Individual
from pymoo.model.repair import Repair
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_crossover, get_mutation
from pymoo.model.mating import Mating

from dask.distributed import Client, LocalCluster

from pymoo.model.initialization import Initialization
from pymoo.model.duplicate import NoDuplicateElimination
from pymoo.model.individual import Individual
from pymoo.operators.sampling.random_sampling import FloatRandomSampling









parser = argparse.ArgumentParser()
parser.add_argument('-p','--ports', nargs='+', type=int, default=[2003, 2006], help='TCP port(s) to listen to (default: 2003 2006)')
parser.add_argument("-s", "--scheduler_port", type=int, default=8785)
parser.add_argument("-d", "--dashboard_address", type=int, default=8786)
parser.add_argument("-r", "--route_type", type=str, default='town05_right_0')
parser.add_argument("-c", "--scenario_type", type=str, default='default')
parser.add_argument('-a','--algorithm_name', type=str, default='nsga2-un')
parser.add_argument("-m", "--ego_car_model", type=str, default='lbc')
parser.add_argument("--has_display", type=str, default='0')
parser.add_argument("--root_folder", type=str, default='run_results')

parser.add_argument("--episode_max_time", type=int, default=50)
parser.add_argument("--n_gen", type=int, default=12)
parser.add_argument("--pop_size", type=int, default=100)
parser.add_argument("--has_run_num", type=int, default=1500)
parser.add_argument("--outer_iterations", type=int, default=3)
parser.add_argument('--objective_weights', nargs='+', type=float, default=[-1, 1, 1, 1, -1])
parser.add_argument('--check_unique_coeff', nargs='+', type=float, default=[0, 0.15, 0.5])

arguments = parser.parse_args()


ports = arguments.ports
scheduler_port = arguments.scheduler_port
dashboard_address = arguments.dashboard_address

# ['town01_left_0', 'town07_front_0', 'town05_front_0', 'town05_right_0']
route_type = arguments.route_type
# ['default', 'leading_car_braking', 'vehicles_only', 'no_static']
scenario_type = arguments.scenario_type
# [random', 'nsga2', 'nsga2-dt', 'nsga2-emcmc', 'nsga2-un', 'nsga2-un-emcmc']
algorithm_name = arguments.algorithm_name
# ['lbc', 'auto_pilot', 'pid_agent']
ego_car_model = arguments.ego_car_model

os.environ['HAS_DISPLAY'] = arguments.has_display
root_folder = arguments.root_folder


episode_max_time = arguments.episode_max_time
global_n_gen = arguments.n_gen

pop_size = arguments.pop_size
# only used when algorithm_name is nsga2-dt
outer_iterations = arguments.outer_iterations

if algorithm_name in ['nsga2-un', 'nsga2-un-emcmc', 'nsga2-dt-un']:
    use_unique_bugs = True
else:
    use_unique_bugs = False

if algorithm_name in ['nsga2-emcmc', 'nsga2-un-emcmc']:
    emcmc = True
else:
    emcmc = False

# [ego_linear_speed, closest_dist, offroad_d, wronglane_d, dev_dist]
# objective_weights = np.array([-1, 1, 1, 1, -1])
objective_weights = np.array(arguments.objective_weights)

check_unique_coeff = arguments.check_unique_coeff

has_run_num = arguments.has_run_num
finish_after_has_run = True




random_seeds = [10, 20, 30]
rng = np.random.default_rng(random_seeds[0])

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

scenario_folder = 'scenario_files'
if not os.path.exists('scenario_files'):
    os.mkdir(scenario_folder)
scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

# This is used to control how this program use GPU
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

resume_run = False
save = True
save_path = 'ga_intermediate.pkl'


max_running_time = 3600*24

# ['generations', 'max_time']
global_termination_condition = 'generations'
default_objectives = [0, 7, 7, 7, 0, 0, 0, 0]















'''
for customizing weather choices, static_types, pedestrian_types, vehicle_types, and vehicle_colors, make changes to object_types.py
'''



class MyProblem(Problem):

    def __init__(self, elementwise_evaluation, bug_parent_folder, non_bug_parent_folder, town_name, scenario, direction, route_str, scenario_file, ego_car_model, scheduler_port, dashboard_address, customized_config, ports=[2000], episode_max_time=10000, customized_parameters_distributions={}, customized_center_transforms={}, call_from_dt=False, dt=False, estimator=None, critical_unique_leaves=None, cumulative_info=None, objective_weights=np.array([0, 0, 1, 1, -1]), check_unique_coeff=[0, 0.15, 0.5]):

        customized_parameters_bounds = customized_config['customized_parameters_bounds']
        customized_parameters_distributions = customized_config['customized_parameters_distributions']
        customized_center_transforms = customized_config['customized_center_transforms']
        customized_constraints = customized_config['customized_constraints']


        self.objective_weights = objective_weights
        self.customized_constraints = customized_constraints

        self.call_from_dt = call_from_dt
        self.dt = dt
        self.estimator = estimator
        self.critical_unique_leaves = critical_unique_leaves


        self.objectives_list = []
        self.x_list = []
        self.y_list = []
        self.F_list = []



        self.scheduler_port = scheduler_port
        self.dashboard_address = dashboard_address
        self.ports = ports
        self.episode_max_time = episode_max_time



        self.bug_folder = bug_parent_folder
        self.non_bug_folder = non_bug_parent_folder


        self.town_name = town_name
        self.scenario = scenario
        self.direction = direction
        self.route_str = route_str
        self.scenario_file = scenario_file
        self.ego_car_model = ego_car_model



        if cumulative_info:
            self.counter = cumulative_info['counter']
            self.has_run = cumulative_info['has_run']
            self.start_time = cumulative_info['start_time']

            self.time_list = cumulative_info['time_list']

            self.bugs = cumulative_info['bugs']
            self.unique_bugs = cumulative_info['unique_bugs']

            self.bugs_type_list = cumulative_info['bugs_type_list']
            self.bugs_inds_list = cumulative_info['bugs_inds_list']

            self.bugs_num_list = cumulative_info['bugs_num_list']
            self.unique_bugs_num_list = cumulative_info['unique_bugs_num_list']
            self.has_run_list = cumulative_info['has_run_list']
        else:
            self.counter = 0
            self.has_run = 0
            self.start_time = time.time()

            self.time_list = []
            self.bugs = []
            self.unique_bugs = []

            self.bugs_type_list = []
            self.bugs_inds_list = []

            self.bugs_num_list = []
            self.unique_bugs_num_list = []
            self.has_run_list = []




        fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels = setup_bounds_mask_labels_distributions_stage1()
        customize_parameters(parameters_min_bounds, customized_parameters_bounds)
        customize_parameters(parameters_max_bounds, customized_parameters_bounds)


        fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels, parameters_distributions, n_var = setup_bounds_mask_labels_distributions_stage2(fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels)
        customize_parameters(parameters_min_bounds, customized_parameters_bounds)
        customize_parameters(parameters_max_bounds, customized_parameters_bounds)
        customize_parameters(parameters_distributions, customized_parameters_distributions)




        for d in [fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds]:
            for k, v in d.items():
                assert not hasattr(self, k), k+'should not appear twice.'
                setattr(self, k, v)


        xl = [pair[1] for pair in parameters_min_bounds.items()]
        xu = [pair[1] for pair in parameters_max_bounds.items()]


        self.parameters_min_bounds = parameters_min_bounds
        self.parameters_max_bounds = parameters_max_bounds
        self.mask = mask
        self.labels = labels
        self.parameters_distributions = parameters_distributions
        self.customized_center_transforms = customized_center_transforms

        self.p, self.c, th = check_unique_coeff
        self.th = int(len(self.labels) * th)
        self.check_unique_coeff = (self.p, self.c, self.th)

        self.launch_server = True

        super().__init__(n_var=n_var, n_obj=4, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=elementwise_evaluation)






    def _evaluate(self, X, out, *args, **kwargs):
        objective_weights = self.objective_weights
        customized_center_transforms = self.customized_center_transforms

        waypoints_num_limit = self.waypoints_num_limit
        num_of_static_max = self.num_of_static_max
        num_of_pedestrians_max = self.num_of_pedestrians_max
        num_of_vehicles_max = self.num_of_vehicles_max

        episode_max_time = self.episode_max_time
        call_from_dt = self.call_from_dt
        bug_folder = self.bug_folder
        non_bug_folder = self.non_bug_folder

        parameters_min_bounds = self.parameters_min_bounds
        parameters_max_bounds = self.parameters_max_bounds
        labels = self.labels
        customized_constraints = self.customized_constraints

        dt = self.dt
        estimator = self.estimator
        critical_unique_leaves = self.critical_unique_leaves

        mean_objectives_across_generations_path = os.path.join(self.bug_folder, 'mean_objectives_across_generations.txt')


        town_name = self.town_name
        scenario = self.scenario
        direction = self.direction
        route_str = self.route_str
        scenario_file = self.scenario_file
        ego_car_model = self.ego_car_model

        all_final_generated_transforms_list = []




        def fun(x, launch_server, counter):
            if (dt and not is_critical_region(x[:-1], estimator, critical_unique_leaves)) or if_violate_constraints(x, customized_constraints, labels):
                objectives = default_objectives
                F = np.array(objectives[:objective_weights.shape[0]]) * objective_weights
                return F, None, None, None, objectives, 0, None

            else:


                # x = denormalize_by_entry(self, x)

                customized_data = convert_x_to_customized_data(x, waypoints_num_limit, num_of_static_max, num_of_pedestrians_max, num_of_vehicles_max, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds)


                # run simulation
                objectives, loc, object_type, info, save_path = run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model)

                print(counter, objectives)

                # [ego_linear_speed, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light]
                F = np.array(objectives[:objective_weights.shape[0]]) * objective_weights


                info = {**info, 'x':x, 'waypoints_num_limit':waypoints_num_limit, 'num_of_static_max':num_of_static_max, 'num_of_pedestrians_max':num_of_pedestrians_max, 'num_of_vehicles_max':num_of_vehicles_max, 'customized_center_transforms':customized_center_transforms,
                'parameters_min_bounds':parameters_min_bounds,
                'parameters_max_bounds':parameters_max_bounds}

                cur_info = {'counter':counter, 'x':x, 'objectives':objectives,  'loc':loc, 'object_type':object_type, 'labels':labels, 'info': info}


                is_bug = check_bug(objectives)

                if is_bug:
                    cur_folder = make_hierarchical_dir([bug_folder, str(counter)])
                else:
                    cur_folder = make_hierarchical_dir([non_bug_folder, str(counter)])

                with open(cur_folder+'/'+'cur_info.pickle', 'wb') as f_out:
                    pickle.dump(cur_info, f_out)

                if is_bug:
                    # copy data to current folder if it is a bug
                    try:
                        new_path = os.path.join(cur_folder, 'data')
                        shutil.copytree(save_path, new_path)
                    except:
                        print('fail to copy from', save_path)

                # hack:
                cur_port = int(x[-1])
                filename = 'tmp_folder/'+str(cur_port)+'.pickle'
                with open(filename, 'rb') as f_in:
                    all_final_generated_transforms = pickle.load(f_in)

                return F, loc, object_type, info, objectives, 1, all_final_generated_transforms




        def submit_and_run_jobs(ind_start, ind_end, launch_server, job_results):
            time_elapsed = 0
            jobs = []
            for i in range(ind_start, ind_end):
                j = i % len(self.ports)
                port = self.ports[j]
                worker = workers[j]
                x = np.concatenate([X[i], np.array([port])])
                jobs.append(client.submit(fun, x, launch_server, self.counter, workers=worker))
                print(i, self.counter)
                self.counter += 1


            for i in range(len(jobs)):
                job = jobs[i]
                F, loc, object_type, info, objectives, has_run, all_final_generated_transforms_i = job.result()
                all_final_generated_transforms_list.append(all_final_generated_transforms_i)

                self.has_run_list.append(has_run)
                self.has_run += has_run
                # record bug
                if check_bug(objectives):
                    bug_str = None
                    if objectives[0] > 0:
                        collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types, 'motercycle_collision':motorcycle_types, 'cyclist_collision':cyclist_types, 'static_collision':static_types}
                        for k,v in collision_types.items():
                            if object_type in v:
                                bug_str = k
                        if not bug_str:
                            bug_str = 'unknown_collision'+'_'+object_type
                        bug_type = 1
                    elif objectives[5]:
                        bug_str = 'offroad'
                        bug_type = 2
                    elif objectives[6]:
                        bug_str = 'wronglane'
                        bug_type = 3
                    else:
                        bug_str = 'unknown'
                        bug_type = 4
                    with open(mean_objectives_across_generations_path, 'a') as f_out:
                        f_out.write(str(i)+','+bug_str+'\n')

                    self.bugs.append(X[i].astype(float))
                    self.bugs_inds_list.append(self.counter-len(jobs)+i)
                    self.bugs_type_list.append(bug_type)

                    self.y_list.append(bug_type)
                else:
                    self.y_list.append(0)
                # we don't want to store port number
                self.x_list.append(X[i])
                self.F_list.append(F)
                self.objectives_list.append(np.array(objectives))
                job_results.append(F)

            # print(all_final_generated_transforms_list)

            # hack:
            with open('tmp_folder/total.pickle', 'wb') as f_out:
                pickle.dump(all_final_generated_transforms_list, f_out)


            # record time elapsed and bug numbers
            self.time_list.append(time_elapsed)



        def process_specific_bug(bug_ind):
            chosen_bugs = np.array(self.bugs_type_list) == bug_ind

            specific_bugs = np.array(self.bugs)[chosen_bugs]
            specific_bugs_inds_list = np.array(self.bugs_inds_list)[chosen_bugs]

            unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(specific_bugs, self.mask, self.xl, self.xu, self.p, self.c, self.th)


            unique_specific_bugs_inds_list = specific_bugs_inds_list[specific_distinct_inds]

            # print(bug_ind, specific_distinct_inds, specific_bugs_inds_list, unique_specific_bugs_inds_list)

            return list(unique_specific_bugs), list(unique_specific_bugs_inds_list), len(unique_specific_bugs)




        job_results = []

        with LocalCluster(scheduler_port=self.scheduler_port, dashboard_address=self.dashboard_address, n_workers=len(self.ports), threads_per_worker=1) as cluster, Client(cluster, connection_limit=8192) as client:
            workers = []
            for k in client.has_what():
                workers.append(k[len('tcp://'):])

            end_ind = np.min([len(self.ports), X.shape[0]])
            print('end_ind, X.shape[0]', end_ind, X.shape[0])
            rng = np.random.default_rng(random_seeds[1])
            submit_and_run_jobs(0, end_ind, self.launch_server, job_results)
            self.launch_server = False
            time_elapsed = time.time() - self.start_time


            if X.shape[0] > len(self.ports):
                rng = np.random.default_rng(random_seeds[2])
                submit_and_run_jobs(end_ind, X.shape[0], self.launch_server, job_results)


            time_elapsed = time.time() - self.start_time
            print('\n'*10)
            print('+'*100)
            mean_objectives_this_generation = np.mean(np.array(self.objectives_list[-X.shape[0]:]), axis=0)




            unique_collision_bugs, unique_collision_bugs_inds_list, unique_collision_num = process_specific_bug(1)
            unique_offroad_bugs, unique_offroad_bugs_inds_list, unique_offroad_num = process_specific_bug(2)
            unique_wronglane_bugs, unique_wronglane_bugs_inds_list, unique_wronglane_num = process_specific_bug(3)

            # print(unique_collision_bugs, unique_offroad_bugs, unique_wronglane_bugs, type(unique_collision_bugs), type(unique_offroad_bugs), type(unique_wronglane_bugs))
            self.unique_bugs = unique_collision_bugs + unique_offroad_bugs + unique_wronglane_bugs
            unique_bugs_inds_list = unique_collision_bugs_inds_list + unique_offroad_bugs_inds_list + unique_wronglane_bugs_inds_list


            num_of_bugs = len(self.bugs)
            num_of_unique_bugs = len(self.unique_bugs)

            self.bugs_num_list.append(num_of_bugs)
            self.unique_bugs_num_list.append(num_of_unique_bugs)



            num_of_collisions = np.sum(np.array(self.bugs_type_list)==1)
            num_of_offroad = np.sum(np.array(self.bugs_type_list)==2)
            num_of_wronglane = np.sum(np.array(self.bugs_type_list)==3)


            # print(unique_collision_bugs, unique_offroad_bugs, unique_wronglane_bugs)
            # print(unique_collision_bugs_inds_list, unique_offroad_bugs_inds_list, unique_wronglane_bugs_inds_list)
            # print(unique_collision_num, unique_offroad_num, unique_wronglane_num)
            # print()

            print(self.counter, self.has_run, time_elapsed, num_of_bugs, num_of_unique_bugs, num_of_collisions, num_of_offroad, num_of_wronglane, mean_objectives_this_generation, unique_collision_num, unique_offroad_num, unique_wronglane_num)
            print(self.bugs_inds_list)
            print(unique_bugs_inds_list)


            for i in range(X.shape[0]-1):
                for j in range(i+1, X.shape[0]):
                    if np.sum(X[i]-X[j])==0:
                        print(X.shape[0], i, j, 'same')

            with open(mean_objectives_across_generations_path, 'a') as f_out:
                f_out.write(','.join([str(x) for x in [self.counter, self.has_run, time_elapsed, num_of_bugs, num_of_unique_bugs, num_of_collisions, num_of_offroad, num_of_wronglane, unique_collision_num, unique_offroad_num, unique_wronglane_num]]+[str(x) for x in mean_objectives_this_generation])+'\n')
                f_out.write(';'.join([str(ind) for ind in unique_bugs_inds_list])+'\n')
            print('+'*100)
            print('\n'*10)
            # os.system('sudo chmod -R 777 '+self.bug_folder)



            # save intermediate results
            if len(self.x_list) > 0:
                X = np.stack(self.x_list)
                F = np.stack(self.F_list)
                objectives = np.stack(self.objectives_list)
            else:
                X = []
                F = []
                objectives = []

            non_dt_save_file = '_'.join([algorithm_name, route_type, scenario_type, ego_car_model, str(global_n_gen), str(pop_size)])
            pth = os.path.join(self.bug_folder, non_dt_save_file)

            np.savez(pth, X=X, y=np.array(self.y_list), F=F, objectives=objectives, time_list=np.array(self.time_list), bugs_num_list=np.array(self.bugs_num_list), unique_bugs_num_list=np.array(self.unique_bugs_num_list), has_run_list=self.has_run_list, labels=self.labels, mask=self.mask, xl=self.xl, xu=self.xu, p=self.p, c=self.c, th=self.th, route_type=route_type, scenario_type=scenario_type)
            print('npz saved')


        out["F"] = np.row_stack(job_results)




def run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model, ego_car_model_path=None, rerun=False, rerun_folder=None):
    arguments = arguments_info()
    arguments.port = customized_data['port']
    arguments.debug = 1
    if rerun:
        arguments.debug = 0



    if ego_car_model == 'lbc':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'
        arguments.agent_config = 'models/epoch=24.ckpt'
        # arguments.agent_config = 'models/stage2_retrain_9_50_leading_car_25_epoch=21.ckpt'
        base_save_folder = 'collected_data_customized'
    elif ego_car_model == 'auto_pilot':
        arguments.agent = 'leaderboard/team_code/auto_pilot.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_autopilot'
    elif ego_car_model == 'pid_agent':
        arguments.agent = 'scenario_runner/team_code/pid_agent.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_pid_agent'
    elif ego_car_model == 'map_model':
        arguments.agent = 'scenario_runner/team_code/map_agent.py'
        arguments.agent_config = 'models/stage1_default_50_epoch=16.ckpt'
        base_save_folder = 'collected_data_map_model'
    else:
        print('unknown ego_car_model:', ego_car_model)

    if ego_car_model_path:
        arguments.agent_config = ego_car_model_path


    if rerun:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, '/rerun', str(int(arguments.port)), str(call_from_dt)])
    else:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, str(int(arguments.port)), str(call_from_dt)])



    arguments.scenarios = scenario_file




    statistics_manager = StatisticsManager()


    # Fixed Hyperparameters
    multi_actors_scenarios = ['Scenario12']
    # sample_factor is an integer between [1, 8]
    sample_factor = 5
    weather_index = customized_data['weather_index']


    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = weather_index
    os.environ['WEATHER_INDEX'] = str(weather_index)

    town_scenario_direction = town_name + '/' + scenario

    folder_1 = os.environ['SAVE_FOLDER'] + '/' + town_name
    if not os.path.exists(folder_1):
        os.mkdir(folder_1)
    folder_2 = folder_1 + '/' + scenario
    if not os.path.exists(folder_2):
        os.mkdir(folder_2)

    if scenario in multi_actors_scenarios:
        town_scenario_direction += '/' + direction
        folder_2 += '/' + direction
        if not os.path.exists(folder_2):
            os.mkdir(folder_2)

    os.environ['SAVE_FOLDER'] = folder_2
    arguments.save_folder = os.environ['SAVE_FOLDER']

    route_prefix = 'leaderboard/data/customized_routes/' + town_scenario_direction + '/route_'

    arguments.routes = route_prefix + route_str + '.xml'
    os.environ['ROUTES'] = arguments.routes

    save_path = os.path.join(arguments.save_folder, 'route_'+route_str)

    # TBD: for convenience
    arguments.deviations_folder = save_path


    # extract waypoints along route
    import xml.etree.ElementTree as ET
    tree = ET.parse(arguments.routes)
    route_waypoints = []



    # this iteration should only go once since we only keep one route per file
    for route in tree.iter("route"):
        route_id = route.attrib['id']
        route_town = route.attrib['town']

        for waypoint in route.iter('waypoint'):
            route_waypoints.append(create_transform(float(waypoint.attrib['x']), float(waypoint.attrib['y']), float(waypoint.attrib['z']), float(waypoint.attrib['pitch']), float(waypoint.attrib['yaw']), float(waypoint.attrib['roll'])))


    # --------------------------------------------------------------------------



    customized_data['using_customized_route_and_scenario'] = True
    customized_data['destination'] = route_waypoints[-1].location
    customized_data['sample_factor'] = sample_factor
    customized_data['number_of_attempts_to_request_actor'] = 10




    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager, launch_server, episode_max_time)
        leaderboard_evaluator.run(arguments, customized_data)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator
        # collect signals for estimating objectives


        objectives, loc, object_type = estimate_objectives(save_path)




    info = {'episode_max_time':episode_max_time,
    'call_from_dt':call_from_dt,
    'town_name':town_name,
    'scenario':scenario,
    'direction':direction,
    'route_str':route_str,
    'route_type':route_type,
    'ego_car_model':ego_car_model}


    if rerun:
        is_bug = check_bug(objectives)
        if is_bug:
            print('\n'*3, 'rerun also causes a bug!!! will not save this', '\n'*3)
            try:
                # use this version to merge into the existing folder
                from distutils.dir_util import copy_tree
                copy_tree(save_path, rerun_folder)
            except:
                print('fail to copy from', save_path)
                traceback.print_exc()
        else:
            assert rerun_folder
            try:
                # use this version to merge into the existing folder
                from distutils.dir_util import copy_tree
                copy_tree(save_path, rerun_folder)
            except:
                print('fail to copy from', save_path)
                traceback.print_exc()


    return objectives, loc, object_type, info, save_path




def estimate_objectives(save_path):
    events_path = os.path.join(save_path, 'events.txt')
    deviations_path = os.path.join(save_path, 'deviations.txt')

    # hack: threshold to avoid too large influence
    ego_linear_speed = 0
    min_d = 7
    offroad_d = 7
    wronglane_d = 7
    dev_dist = 0

    is_offroad = 0
    is_wrong_lane = 0
    is_run_red_light = 0


    with open(deviations_path, 'r') as f_in:
        for line in f_in:
            type, d = line.split(',')
            d = float(d)
            if type == 'min_d':
                min_d = np.min([min_d, d])
            elif type == 'offroad_d':
                offroad_d = np.min([offroad_d, d])
            elif type == 'wronglane_d':
                wronglane_d = np.min([wronglane_d, d])
            elif type == 'dev_dist':
                dev_dist = np.max([dev_dist, d])





    x = None
    y = None
    object_type = None

    infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'off_road']

    try:
        with open(events_path) as json_file:
            events = json.load(json_file)
    except:
        print('events_path', events_path, 'is not found')
        return default_objectives, (None, None), None

    infractions = events['_checkpoint']['records'][0]['infractions']
    status = events['_checkpoint']['records'][0]['status']

    for infraction_type in infraction_types:
        for infraction in infractions[infraction_type]:
            if 'collisions' in infraction_type:
                typ = re.search('.*with type=(.*) and id.*', infraction)
                print(infraction, typ)
                if typ:
                    object_type = typ.group(1)
                loc = re.search('.*x=(.*), y=(.*), z=(.*), ego_linear_speed=(.*), other_actor_linear_speed=(.*)\)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    ego_linear_speed = float(loc.group(4))
                    other_actor_linear_speed = float(loc.group(5))

            elif infraction_type == 'off_road':
                loc = re.search('.*x=(.*), y=(.*), z=(.*)\)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    is_offroad = 1
            else:
                if infraction_type == 'wrong_lane':
                    is_wrong_lane = 1
                elif infraction_type == 'red_light':
                    is_run_red_light = 1
                loc = re.search('.*x=(.*), y=(.*), z=(.*)[\),]', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))

    # limit impact of too large values
    ego_linear_speed = np.min([ego_linear_speed, 7])
    dev_dist = np.min([dev_dist, 7])

    return [ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light], (x, y), object_type



class MySampling(Sampling):
    '''
    dimension correspondence

    Define:
    n1=problem.waypoints_num_limit
    n2=problem.num_of_static_max
    n3=problem.num_of_pedestrians_max
    n4=problem.num_of_vehicles_max

    global
    0: friction, real, [0, 1].
    1: weather_index, int, [0, problem.num_of_weathers].
    2: num_of_static, int, [0, n2].
    3: num_of_pedestrians, int, [0, n3].
    4: num_of_vehicles, int, [0, n4].

    ego-car
    5 ~ 4+n1*2: waypoints perturbation [(dx_i, dy_i)] with length n1.
    dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].


    static
    5+n1*2 ~ 4+n1*2+n2*4: [(static_type_i, x w.r.t. center, y w.r.t. center, yaw)] with length n2.
    static_type_i, int, [0, problem.num_of_static_types).
    x_i, real, [problem.static_x_min, problem.static_x_max].
    y_i, real, [problem.static_y_min, problem.static_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).

    pedestrians
    5+n1*2+n2*4 ~ 4+n1*2+n2*4+n3*7: [(pedestrian_type_i, x_i, y_i, yaw_i, trigger_distance_i, speed_i, dist_to_travel_i)] with length n3.
    pedestrian_type_i, int, [0, problem.num_of_static_types)
    x_i, real, [problem.pedestrian_x_min, problem.pedestrian_x_max].
    y_i, real, [problem.pedestrian_y_min, problem.pedestrian_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).
    trigger_distance_i, real, [problem.pedestrian_trigger_distance_min, problem.pedestrian_trigger_distance_max].
    speed_i, real, [problem.pedestrian_speed_min, problem.pedestrian_speed_max].
    dist_to_travel_i, real, [problem.pedestrian_dist_to_travel_min, problem.pedestrian_dist_to_travel_max].

    vehicles
    5+n1*2+n2*4+n3*7 ~ 4+n1*2+n2*4+n3*7+n4*(14+n1*2): [(vehicle_type_i, x_i, y_i, yaw_i, initial_speed_i, trigger_distance_i, targeted_speed_i, waypoint_follower_i, targeted_x_i, targeted_y_i, avoid_collision_i, dist_to_travel_i, target_yaw_i, color_i, [(dx_i, dy_i)] with length n1)] with length n4.
    vehicle_type_i, int, [0, problem.num_of_vehicle_types)
    x_i, real, [problem.vehicle_x_min, problem.vehicle_x_max].
    y_i, real, [problem.vehicle_y_min, problem.vehicle_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).
    initial_speed_i, real, [problem.vehicle_initial_speed_min, problem.vehicle_initial_speed_max].
    trigger_distance_i, real, [problem.vehicle_trigger_distance_min, problem.vehicle_trigger_distance_max].
    targeted_speed_i, real, [problem.vehicle_targeted_speed_min, problem.vehicle_targeted_speed_max].
    waypoint_follower_i, boolean, [0, 1]
    targeted_x_i, real, [problem.targeted_x_min, problem.targeted_x_max].
    targeted_y_i, real, [problem.targeted_y_min, problem.targeted_y_max].
    avoid_collision_i, boolean, [0, 1]
    dist_to_travel_i, real, [problem.vehicle_dist_to_travel_min, problem.vehicle_dist_to_travel_max].
    target_yaw_i, real, [problem.yaw_min, problem.yaw_max).
    color_i, int, [0, problem.num_of_vehicle_colors).
    dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].


    '''
    def __init__(self, use_unique_bugs, check_unique_coeff):
        self.use_unique_bugs = use_unique_bugs
        self.check_unique_coeff = check_unique_coeff
        assert len(self.check_unique_coeff) == 3

    def _do(self, problem, n_samples, **kwargs):
        p, c, th = self.check_unique_coeff
        xl = problem.xl
        xu = problem.xu
        mask = problem.mask
        labels = problem.labels
        parameters_distributions = problem.parameters_distributions
        max_sample_times = n_samples*500



        X = []
        def subroutine(disable_unique_bugs=False):
            sample_time = 0
            while sample_time < max_sample_times and len(X) < n_samples:
                sample_time += 1
                x = []
                for i, dist in enumerate(parameters_distributions):
                    typ = mask[i]
                    lower = xl[i]
                    upper = xu[i]
                    assert lower <= upper, problem.labels[i]+','+str(lower)+'>'+str(upper)
                    label = labels[i]
                    if typ == 'int':
                        val = rng.integers(lower, upper+1)
                    elif typ == 'real':
                        if dist[0] == 'normal':
                            if dist[1] == None:
                                mean = (lower+upper)/2
                            else:
                                mean = dist[1]
                            val = rng.normal(mean, dist[2], 1)[0]
                        else: # default is uniform
                            val = rand_real(rng, lower, upper)
                        val = np.clip(val, lower, upper)
                    x.append(val)
                # print(if_violate_constraints(x, problem.customized_constraints, problem.labels), use_unique_bugs, is_distinct(x, X, mask, xl, xu, p, c, th))
                if not if_violate_constraints(x, problem.customized_constraints, problem.labels) and (disable_unique_bugs or not self.use_unique_bugs or is_distinct(x, X, mask, xl, xu, p, c, th)):
                    x = np.array(x).astype(float)
                    X.append(x)

            return sample_time

        sample_time_1 = subroutine()
        print(len(X), 'samples after first round of sampling')
        sample_time_2 = subroutine(True)
        print(len(X), 'samples after second round of sampling')

        X = np.stack(X)

        print('\n'*3, 'We sampled', X.shape[0], '/', n_samples, 'samples', 'by sampling', sample_time_1+sample_time_2, 'times' '\n'*3)
        return X



def do_emcmc(parents, off, n_gen, objective_weights):
    base_val = np.sum(np.array(default_objectives[:len(objective_weights)])*np.array(objective_weights))
    filtered_off = []
    F_list = []
    for i in off:
        for p in parents:
            print(i.F, p.F)

            i_val = np.sum(np.array(i.F) * np.array(objective_weights))
            p_val = np.sum(np.array(p.F) * np.array(objective_weights))

            print('1', base_val, i_val, p_val)
            i_val = np.abs(base_val-i_val)
            p_val = np.abs(base_val-p_val)
            prob = np.min([i_val / p_val, 1])
            print('2', base_val, i_val, p_val, prob)

            if np.random.uniform() < prob:
                filtered_off.append(i.X)
                F_list.append(i.F)

    pop = Population(len(filtered_off), individual=Individual())
    pop.set("X", filtered_off, "F", F_list, "n_gen", n_gen, "CV", [0 for _ in range(len(filtered_off))], "feasible", [[True] for _ in range(len(filtered_off))])

    return Population.merge(parents, off)


class MyMating(Mating):
    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 use_unique_bugs,
                 emcmc,
                 **kwargs):

        super().__init__(selection, crossover, mutation, **kwargs)
        self.use_unique_bugs = use_unique_bugs
        self.emcmc = emcmc

    def do(self, problem, pop, n_offsprings, **kwargs):

        # the population object to be used
        off = pop.new()
        parents = pop.new()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:
            print('n_infills', n_infills)
            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off, _parents = self._do(problem, pop, n_remaining, **kwargs)


            # repair the individuals if necessary - disabled if repair is NoRepair
            _off = self.repair.do(problem, _off, **kwargs)

            # eliminate the duplicates - disabled if it is NoRepair
            if self.use_unique_bugs:
                _off, no_duplicate, _ = self.eliminate_duplicates.do(_off, problem.unique_bugs, off, return_indices=True, to_itself=True)
                _parents = _parents[no_duplicate]
                assert len(_parents)==len(_off)


            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                # IMPORTANT: Interestingly, this makes a difference in performance
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]
                _parents = _parents[:n_remaining]


            # add to the offsprings and increase the mating counter
            off = Population.merge(off, _off)
            parents = Population.merge(parents, _parents)
            n_infills += 1

            # if no new offsprings can be generated within a pre-specified number of generations
            if n_infills > self.n_max_iterations:
                break

        assert len(parents)==len(off)

        return off, parents



    # only to get parents
    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)
            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)

            parents_obj = pop[parents].reshape([-1, 1]).squeeze()
        else:
            parents_obj = parents


        # do the crossover using the parents index and the population - additional data provided if necessary
        _off = self.crossover.do(problem, pop, parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        _off = self.mutation.do(problem, _off, **kwargs)


        return _off, parents_obj



class NSGA2_DT(NSGA2):
    def __init__(self, dt=False, X=None, F=None, emcmc=False, plain_sampling=None, sampling=None, algorithm_name='nsga2-un', **kwargs):
        self.dt = dt
        self.X = X
        self.F = F
        self.emcmc = emcmc
        self.algorithm_name = algorithm_name
        self.sampling = sampling

        super().__init__(**kwargs)

        self.plain_initialization = Initialization(plain_sampling, individual=Individual(), repair=self.repair, eliminate_duplicates= NoDuplicateElimination())


        # heuristic: we keep up about 2 times of each generation's population
        self.survival_size = self.pop_size * 2

    # mainly used to modify survival
    def _next(self):
        if self.algorithm_name == 'random':
            self.off = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        else:
            # do the mating using the current population
            self.off, parents = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

            print('\n'*3, 'len 0', len(self.off), '\n'*3)

            if len(self.off) < self.n_offsprings:
                remaining_num = self.n_offsprings - len(self.off)
                remaining_off = self.initialization.do(self.problem, remaining_num, algorithm=self)
                remaining_parrents = remaining_off

                self.off = Population.merge(self.off, remaining_off)
                parents = Population.merge(parents, remaining_parrents)

                print('\n'*3, 'len 1', len(self.off), '\n'*3)

            if len(self.off) < self.n_offsprings:
                remaining_num = self.n_offsprings - len(self.off)
                remaining_off = self.plain_initialization.do(self.problem, remaining_num, algorithm=self)
                remaining_parrents = remaining_off

                self.off = Population.merge(self.off, remaining_off)
                parents = Population.merge(parents, remaining_parrents)

                print('\n'*3, 'len 2', len(self.off), '\n'*3)



        self.off.set("n_gen", self.n_gen)
        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0 or (not self.problem.call_from_dt and finish_after_has_run and self.problem.has_run >= has_run_num):
            self.termination.force_termination = True
            print("Mating cannot generate new springs, terminate earlier.")
            return

        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")


        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)



        if self.algorithm_name == 'random':
            self.pop = self.off
        elif self.emcmc:
            new_pop = do_emcmc(parents, self.off, self.n_gen, self.problem.objective_weights)

            self.pop = Population.merge(self.pop, new_pop)

            if self.survival:
                self.pop = self.survival.do(self.problem, self.pop, self.survival_size, algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)

        else:
            # merge the offsprings with the current population
            self.pop = Population.merge(self.pop, self.off)

            # the do survival selection
            if self.survival:
                self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)




    def _initialize(self):
        if self.dt:
            X_list = list(self.X)
            F_list = list(self.F)
            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])


        else:
            # create the initial population
            pop = Population(0, individual=self.individual)
            pop = self.sampling.do(self.problem, pop_size, pop=pop, algorithm=self)
            pop = self.repair.do(self.problem, pop, algorithm=self)

            if len(pop) < self.pop_size:
                remaining_num = self.pop_size - len(pop)
                remaining_pop = self.plain_initialization.do(self.problem, remaining_num, algorithm=self)
                pop = Population.merge(pop, remaining_pop)
            pop.set("n_gen", self.n_gen)




        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)

        self.pop, self.off = pop, pop









class ClipRepair(Repair):
    """
    A dummy class which can be used to simply do no repair.
    """

    def do(self, problem, pop, **kwargs):
        for i in range(len(pop)):
            pop[i].X = np.clip(pop[i].X, np.array(problem.xl), np.array(problem.xu))
        return pop


class SimpleDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, mask, xu, xl, check_unique_coeff, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask = np.array(mask)
        self.xu = np.array(xu)
        self.xl = np.array(xl)
        self.cmp = lambda a, b: self.is_equal(a, b)
        self.check_unique_coeff = check_unique_coeff
        assert len(self.check_unique_coeff) == 3

    def is_equal(self, a, b):
        if type(b).__module__ == np.__name__:
            b_X = b
        else:
            b_X = b.X
        p, c, th = self.check_unique_coeff
        return is_similar(a.X, b_X, self.mask, self.xl, self.xu, p, c, th)




class MyEvaluator(Evaluator):
    def _eval(self, problem, pop, **kwargs):

        super()._eval(problem, pop, **kwargs)
        # print(pop[0].X)
        # hack:
        label_to_id = {label:i for i, label in enumerate(problem.labels)}

        def correct_spawn_locations(all_final_generated_transforms_list_i, i, object_type, keys):
            object_type_plural = object_type
            if object_type in ['pedestrian', 'vehicle']:
                object_type_plural += 's'

            num_of_objects_ind = label_to_id['num_of_'+object_type_plural]
            pop[i].X[num_of_objects_ind] = 0

            empty_slots = deque()
            for j, (x, y, yaw) in enumerate(all_final_generated_transforms_list_i[object_type]):
                if x == None:
                    empty_slots.append(j)
                else:
                    pop[i].X[num_of_objects_ind] += 1
                    if object_type+'_x_'+str(j) not in label_to_id:
                        print(object_type+'_x_'+str(j))
                        print(all_final_generated_transforms_list_i[object_type])
                        raise
                    x_j_ind = label_to_id[object_type+'_x_'+str(j)]
                    y_j_ind = label_to_id[object_type+'_y_'+str(j)]
                    yaw_j_ind = label_to_id[object_type+'_yaw_'+str(j)]


                    # print(object_type, j)
                    # print('x', pop[i].X[x_j_ind], '->', x)
                    # print('y', pop[i].X[y_j_ind], '->', y)
                    # print('yaw', pop[i].X[yaw_j_ind], '->', yaw)
                    pop[i].X[x_j_ind] = x
                    pop[i].X[y_j_ind] = y
                    pop[i].X[yaw_j_ind] = yaw

                    if len(empty_slots) > 0:
                        q = empty_slots.popleft()
                        print('shift', j, 'to', q)
                        for k in keys:
                            print(k)
                            ind_to = label_to_id[k+'_'+str(q)]
                            ind_from = label_to_id[k+'_'+str(j)]
                            pop[i].X[ind_to] = pop[i].X[ind_from]
                        if object_type == 'vehicle':
                            for p in range(waypoints_num_limit):
                                for waypoint_label in waypoint_labels:
                                    ind_to = label_to_id['_'.join(['vehicle', str(q), waypoint_label, str(p)])]
                                    ind_from = label_to_id['_'.join(['vehicle', str(j), waypoint_label, str(p)])]
                                    pop[i].X[ind_to] = pop[i].X[ind_from]

                        empty_slots.append(j)
            # print()


        with open('tmp_folder/total.pickle', 'rb') as f_in:
            all_final_generated_transforms_list = pickle.load(f_in)

        for i, all_final_generated_transforms_list_i in enumerate(all_final_generated_transforms_list):
            if all_final_generated_transforms_list_i:
                # print(i)
                correct_spawn_locations(all_final_generated_transforms_list_i, i, 'static', static_general_labels)
                correct_spawn_locations(all_final_generated_transforms_list_i, i, 'pedestrian', pedestrian_general_labels)
                correct_spawn_locations(all_final_generated_transforms_list_i, i, 'vehicle', vehicle_general_labels)
                # print('\n'*3)
        # print(pop[0].X)






def customized_minimize(problem,
             algorithm,
             resume_run,
             termination=None,
             **kwargs):
    # create a copy of the algorithm object to ensure no side-effects
    algorithm = copy.deepcopy(algorithm)

    # get the termination if provided as a tuple - create an object
    if termination is not None and not isinstance(termination, Termination):
        if isinstance(termination, str):
            termination = get_termination(termination)
        else:
            termination = get_termination(*termination)


    # initialize the algorithm object given a problem
    algorithm.initialize(problem, termination=termination, **kwargs)

    # if no termination could be found add the default termination either for single or multi objective
    if algorithm.termination is None:
        if problem.n_obj > 1:
            algorithm.termination = MultiObjectiveDefaultTermination()
        else:
            algorithm.termination = SingleObjectiveDefaultTermination()

    # actually execute the algorithm
    res = algorithm.solve()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm

    return res



def run_nsga2_dt():
    end_when_no_critical_region = True
    cumulative_info = None

    X_filtered = None
    F_filtered = None
    X = None
    y = None
    F = None
    objectives = None
    labels = None
    hv = None
    estimator = None
    critical_unique_leaves = None


    now = datetime.now()
    dt_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


    for i in range(outer_iterations):
        dt_time_str_i = dt_time_str
        dt = True
        if i == 0 or np.sum(y)==0:
            dt = False
        X_new, y_new, F_new, objectives_new, labels, hv_new, parent_folder, cumulative_info = run_ga(True, dt, X_filtered, F_filtered, estimator, critical_unique_leaves, dt_time_str_i, i, cumulative_info)

        if finish_after_has_run and cumulative_info['has_run'] > has_run_num:
            break

        if len(X_new) == 0:
            break

        if i == 0:
            X = X_new
            y = y_new
            F = F_new
            objectives = objectives_new
            hv = hv_new

        else:
            X = np.concatenate([X, X_new])
            y = np.concatenate([y, y_new])
            F = np.concatenate([F, F_new])
            objectives = np.concatenate([objectives, objectives_new])
            hv = np.concatenate([hv, hv_new])



        estimator, inds, critical_unique_leaves = filter_critical_regions(X, y)
        X_filtered = X[inds]
        F_filtered = F[inds]

        if len(X_filtered) == 0 and end_when_no_critical_region:
            break


    # Save data
    has_run_list = cumulative_info['has_run_list']
    time_list = cumulative_info['time_list']
    bugs_num_list = cumulative_info['bugs_num_list']
    unique_bugs_num_list = cumulative_info['unique_bugs_num_list']


    n_gen = global_n_gen

    dt_save_file = '_'.join([algorithm_name, route_type, scenario_type, ego_car_model, str(n_gen), str(pop_size), str(outer_iterations), dt_time_str])

    pth = os.path.join(parent_folder, dt_save_file)
    np.savez(pth, X=X, y=y, F=F, objectives=objectives, time_list=time_list, bugs_num_list=bugs_num_list, unique_bugs_num_list=unique_bugs_num_list, labels=labels, hv=hv, has_run_list=has_run_list, route_type=route_type, scenario_type=scenario_type)

    print('npz saved')





def run_ga(call_from_dt=False, dt=False, X=None, F=None, estimator=None, critical_unique_leaves=None, dt_time_str=None, dt_iter=None, cumulative_info=None):

    n_gen = global_n_gen

    if call_from_dt:
        termination_condition = 'generations'
        if dt and len(list(X)) == 0:
            print('No critical leaves!!! Start from random sampling!!!')
            dt = False

        if dt:
            n_gen += 1
    else:
        termination_condition = global_termination_condition









    # scenario_type = 'leading_car_braking'
    customized_d = customized_bounds_and_distributions[scenario_type]
    route_info = customized_routes[route_type]

    town_name = route_info['town_name']
    scenario = 'Scenario12' # This is only for compatibility purpose
    direction = route_info['direction']
    route = route_info['route_id']
    location_list = route_info['location_list']

    route_str = str(route)
    if route < 10:
        route_str = '0'+route_str

    parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file)


    if call_from_dt:
        time_str = dt_time_str
    else:
        now = datetime.now()
        time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


    cur_parent_folder = make_hierarchical_dir([root_folder, algorithm_name, route_type, scenario_type, ego_car_model, time_str])

    if call_from_dt:
        parent_folder = make_hierarchical_dir([cur_parent_folder, str(dt_iter)])
    else:
        parent_folder = cur_parent_folder

    bug_parent_folder = make_hierarchical_dir([parent_folder, 'bugs'])
    non_bug_parent_folder = make_hierarchical_dir([parent_folder, 'non_bugs'])




    if resume_run:
        with open(save_path, 'rb') as f_in:
            problem = pickle.load(f_in)

    else:
        problem = MyProblem(elementwise_evaluation=False, bug_parent_folder=bug_parent_folder, non_bug_parent_folder=non_bug_parent_folder, town_name=town_name, scenario=scenario, direction=direction, route_str=route_str, scenario_file=scenario_file, ego_car_model=ego_car_model, scheduler_port=scheduler_port, dashboard_address=dashboard_address, customized_config=customized_d, ports=ports, episode_max_time=episode_max_time,
        call_from_dt=call_from_dt, dt=dt, estimator=estimator, critical_unique_leaves=critical_unique_leaves, cumulative_info=cumulative_info, objective_weights=objective_weights,
        check_unique_coeff=check_unique_coeff)




    # deal with real and int separately
    crossover = MixedVariableCrossover(problem.mask, {
        "real": get_crossover("real_sbx", prob=0.8, eta=5),
        "int": get_crossover("int_sbx", prob=0.8, eta=5)
    })

    mutation = MixedVariableMutation(problem.mask, {
        "real": get_mutation("real_pm", eta=5, prob=int(0.05*problem.n_var)),
        "int": get_mutation("int_pm", eta=5, prob=int(0.05*problem.n_var))
    })



    selection = TournamentSelection(func_comp=binary_tournament)
    repair = ClipRepair()

    if use_unique_bugs:
        eliminate_duplicates = SimpleDuplicateElimination(mask=problem.mask, xu=problem.xu, xl=problem.xl, check_unique_coeff=problem.check_unique_coeff)
    else:
        eliminate_duplicates = NoDuplicateElimination()

    mating = MyMating(selection,
                    crossover,
                    mutation,
                    use_unique_bugs,
                    emcmc,
                    repair=repair,
                    eliminate_duplicates=eliminate_duplicates,
                    n_max_iterations=100)


    sampling = MySampling(use_unique_bugs=use_unique_bugs, check_unique_coeff=problem.check_unique_coeff)

    plain_sampling = MySampling(use_unique_bugs=False, check_unique_coeff=problem.check_unique_coeff)

    # TBD: customize mutation and crossover to better fit our problem. e.g.
    # might deal with int and real separately
    algorithm = NSGA2_DT(dt=dt, X=X, F=F, emcmc=emcmc, plain_sampling=plain_sampling, algorithm_name=algorithm_name,
                      pop_size=pop_size,
                      sampling=sampling,
                      crossover=crossover,
                      mutation=mutation,
                      eliminate_duplicates=eliminate_duplicates,
                      repair=repair,
                      mating=mating)





    if termination_condition == 'generations':
        termination = ('n_gen', n_gen)
    elif termination_condition == 'max_time':
        termination = ('time', max_running_time)
    else:
        termination = ('n_gen', n_gen)


    # close simulator(s)
    atexit.register(exit_handler, ports, problem.bug_folder, scenario_file)

    # TypeError: can't pickle _asyncio.Task objects when save_history = True
    # verbose has to be set to False to avoid an error when algorithm_name=random
    res = customized_minimize(problem,
                   algorithm,
                   resume_run,
                   termination=termination,
                   seed=0,
                   verbose=False,
                   save_history=False,
                   evaluator=MyEvaluator())

    print('We have found', len(problem.bugs), 'bugs in total.')


    # print("Best solution found: %s" % res.X)
    # print("Function value: %s" % res.F)
    # print("Constraint violation: %s" % res.CV)

    # for drawing hv
    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point=np.array([7.0, 7.0, 7.0, 7.0, 7.0]))
    # collect the population in each generation
    pop_each_gen = [a.pop for a in res.history]
    # receive the population in each generation
    obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]
    # calculate for each generation the HV metric
    hv = np.array([metric.calc(f) for f in obj_and_feasible_each_gen])



    if len(problem.x_list) > 0:
        X = np.stack(problem.x_list)
        F = np.stack(problem.F_list)
        objectives = np.stack(problem.objectives_list)
    else:
        X = []
        F = []
        objectives = []
    y = np.array(problem.y_list)
    time_list = np.array(problem.time_list)
    bugs_num_list = np.array(problem.bugs_num_list)
    unique_bugs_num_list = np.array(problem.unique_bugs_num_list)
    labels = problem.labels
    has_run = problem.has_run
    has_run_list = problem.has_run_list

    mask = problem.mask
    xl = problem.xl
    xu = problem.xu
    p = problem.p
    c = problem.c
    th = problem.th

    # save another data npz for easy comparison with dt results


    non_dt_save_file = '_'.join([algorithm_name, route_type, scenario_type, ego_car_model, str(n_gen), str(pop_size)])
    pth = os.path.join(bug_parent_folder, non_dt_save_file)

    np.savez(pth, X=X, y=y, F=F, objectives=objectives, time_list=time_list, bugs_num_list=bugs_num_list, unique_bugs_num_list=unique_bugs_num_list, has_run=has_run, has_run_list=has_run_list, labels=labels, hv=hv, mask=mask, xl=xl, xu=xu, p=p, c=c, th=th, route_type=route_type, scenario_type=scenario_type)
    print('npz saved')


    if save:
        with open(save_path, 'wb') as f_out:
            pickle.dump(problem, f_out)
            print('-'*100, 'pickled')


    cumulative_info = {
        'has_run': problem.has_run,
        'start_time': problem.start_time,
        'counter': problem.counter,
        'time_list': problem.time_list,
        'bugs': problem.bugs,
        'unique_bugs': problem.unique_bugs,
        'bugs_type_list': problem.bugs_type_list,
        'bugs_inds_list': problem.bugs_inds_list,
        'bugs_num_list': problem.bugs_num_list,
        'unique_bugs_num_list': problem.unique_bugs_num_list,
        'has_run_list': problem.has_run_list
    }


    return X, y, F, objectives, labels, hv, cur_parent_folder, cumulative_info

if __name__ == '__main__':
    if algorithm_name == 'nsga2-dt':
        run_nsga2_dt()
    else:
        run_ga()

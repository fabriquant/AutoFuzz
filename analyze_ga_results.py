import sys
import os
sys.path.append('pymoo')




import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
from dt import filter_critical_regions
from customized_utils import  get_distinct_data_points, check_bug
from ga_fuzzing import default_objectives

def draw_hv(bug_res_path, save_folder):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)
    hv = res['hv']
    n_evals = res['n_evals'].tolist()

    # hv = [0] + hv
    # n_evals = [0] + n_evals


    # visualze the convergence curve
    plt.plot(n_evals, hv, '-o')
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(os.path.join(save_folder, 'hv_across_generations'))
    plt.close()



def draw_performance(bug_res_path, save_folder):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)

    time_bug_num_list = res['time_bug_num_list']

    t_list = []
    n_list = []
    for t, n in time_bug_num_list:
        t_list.append(t)
        n_list.append(n)
    print(t_list)
    print(n_list)
    plt.plot(t_list, n_list, '-o')
    plt.title("Time VS Number of Bugs")
    plt.xlabel("Time")
    plt.ylabel("Number of Bugs")
    plt.savefig(os.path.join(save_folder, 'bug_num_across_time'))
    plt.close()


def analyze_causes(folder, save_folder, total_num, pop_size):



    avg_f = [0 for _ in range(int(total_num // pop_size))]

    causes_list = []
    counter = 0
    for sub_folder_name in os.listdir(folder):
        sub_folder = os.path.join(folder, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".npz"):
                    filepath = os.path.join(sub_folder, filename)
                    bug = np.load(filepath, allow_pickle=True)['bug'][()]

                    ego_linear_speed = float(bug['ego_linear_speed'])
                    causes_list.append((sub_folder_name, ego_linear_speed, bug['offroad_dist'], bug['is_wrong_lane'], bug['is_run_red_light'], bug['status'], bug['loc'], bug['object_type']))

                    ind = int(int(sub_folder_name) // pop_size)
                    avg_f[ind] += (ego_linear_speed / pop_size)*-1

    causes_list = sorted(causes_list, key=lambda t: int(t[0]))
    for c in causes_list:
        print(c)
    print(avg_f)

    plt.plot(np.arange(len(avg_f)), avg_f)
    plt.title("average objective value across generations")
    plt.xlabel("Generations")
    plt.ylabel("average objective value")
    plt.savefig(os.path.join(save_folder, 'f_across_generations'))

    plt.close()

def show_gen_f(bug_res_path):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)

    val = res['val']
    plt.plot(np.arange(len(val)), val)
    plt.show()

def plot_each_bug_num_and_objective_num_over_generations(generation_data_paths):
    # X=X, y=y, F=F, objectives=objectives, time=time_list, bug_num=bug_num_list, labels=labels, hv=hv
    pop_size = 100
    data_list = []
    for generation_data_path in generation_data_paths:
        data = []
        with open(generation_data_path[1], 'r') as f_in:
            for line in f_in:
                tokens = line.split(',')
                if len(tokens) == 2:
                    pass
                else:
                    tokens = [float(x.strip('\n')) for x in line.split(',')]
                    num, has_run, time, bugs, collisions, offroad_num, wronglane_num, speed, min_d, offroad, wronglane, dev = tokens[:12]
                    out_of_road = offroad_num + wronglane_num
                    data.append(np.array([num/pop_size, has_run, time, bugs, collisions, offroad_num, wronglane_num, out_of_road, speed, min_d, offroad, wronglane, dev]))

        data = np.stack(data)
        data_list.append(data)

    labels = [generation_data_paths[i][0] for i in range(len(data_list))]
    data = np.concatenate([data_list[1], data_list[2]], axis=0)

    for i in range(len(data_list[1]), len(data_list[1])+len(data_list[2])):
        data[i] += data_list[1][-1]
    data_list.append(data)

    labels.append('collision+out-of-road')

    fig = plt.figure(figsize=(15, 9))


    plt.suptitle("values over time", fontsize=14)


    info = [(1, 3, 'Bug Numbers'), (6, 4, 'Collision Numbers'), (7, 5, 'Offroad Numbers'), (8, 6, 'Wronglane Numbers'), (9, 7, 'Out-of-road Numbers'), (11, 8, 'Collision Speed'), (12, 9, 'Min object distance'), (13, 10, 'Offroad Directed Distance'), (14, 11, 'Wronglane Directed Distance'), (15, 12, 'Max Deviation')]

    for loc, ind, ylabel in info:
        ax = fig.add_subplot(3, 5, loc)
        for i in [0, 3, 1, 2]:
            if loc < 11 or i < 3:
                label = labels[i]
                if loc >= 11:
                    y = []
                    for j in range(data_list[i].shape[0]):
                        y.append(np.mean(data_list[i][:j+1, ind]))
                else:
                    y = data_list[i][:, ind]
                ax.plot(data_list[i][:, 0], y, label=label)
        if loc == 1:
            ax.legend()
        plt.xlabel("Generations")
        plt.ylabel(ylabel)
    plt.savefig('bug_num_and_objective_num_over_generations')







# list bug types and their run numbers
def list_bug_categories_with_numbers(folder_path):
    l = []
    for sub_folder_name in os.listdir(folder_path):
        sub_folder = os.path.join(folder_path, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".npz"):
                    filepath = os.path.join(sub_folder, filename)
                    bug = np.load(filepath, allow_pickle=True)['bug'][()]
                    if bug['ego_linear_speed'] > 0:
                        cause_str = 'collision'
                    elif bug['is_offroad']:
                        cause_str = 'offroad'
                    elif bug['is_wrong_lane']:
                        cause_str = 'wronglane'
                    else:
                        cause_str = 'unknown'
                    l.append((sub_folder_name, cause_str))


    for n,s in sorted(l, key=lambda t: int(t[0])):
        print(n,s)



# list pickled data
def analyze_data(pickle_path):
    with open(pickle_path, 'rb') as f_out:
        d = pickle.load(f_out)
        X = d['X']
        y = d['y']
        F = d['F']
        objectives = d['objectives']
        print(np.sum(X[10,:]-X[11,:]))
        filter_critical_regions(X, y)
        # TBD: tree diversity



def unique_bug_num(all_X, all_y, mask, xl, xu, cutoff):
    if cutoff == 0:
        return 0, []
    X = all_X[:cutoff]
    y = all_y[:cutoff]

    bug_inds = np.where(y>0)
    bugs = X[bug_inds]


    p = 0
    c = 0.15
    th = int(len(mask)*0.5)

    # TBD: count different bugs separately
    filtered_bugs, unique_bug_inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)


    print(cutoff, len(filtered_bugs), len(bugs))
    return len(filtered_bugs), np.array(unique_bug_inds), bug_inds

# plot two tsne plots for bugs VS normal and data points across generations
def apply_tsne(path, n_gen, pop_size):
    d = np.load(path, allow_pickle=True)
    X = d['X']
    y = d['y']
    mask = d['mask']
    xl = d['xl']
    xu = d['xu']


    cutoff = n_gen * pop_size
    _, unique_bug_inds, bug_inds = unique_bug_num(X, y, mask, xl, xu, cutoff)

    y[bug_inds] = 1
    y[unique_bug_inds] = 2


    generations = []
    for i in range(n_gen):
        generations += [i for _ in range(pop_size)]


    X_embedded = TSNE(n_components=2).fit_transform(X)

    fig = plt.figure(figsize=(9, 9))
    plt.suptitle("tSNE of bugs and unique bugs", fontsize=14)
    ax = fig.add_subplot(111)
    scatter_bug = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap=plt.cm.rainbow)
    plt.title("bugs VS normal")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(handles=scatter_bug.legend_elements()[0], labels=['normal', 'bugs'])


    # fig = plt.figure(figsize=(18, 9))
    #
    # plt.suptitle("tSNE of sampled/generated data points", fontsize=14)
    #
    #
    # ax = fig.add_subplot(121)
    # scatter_bug = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap=plt.cm.rainbow)
    # plt.title("bugs VS normal")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    # plt.legend(handles=scatter_bug.legend_elements()[0], labels=['normal', 'bugs'])
    #
    # ax = fig.add_subplot(122)
    # scatter_gen = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=generations, cmap=plt.cm.rainbow)
    # plt.title("different generations")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    # plt.legend(handles=scatter_gen.legend_elements()[0], labels=[str(i) for i in range(n_gen)])

    plt.savefig('tsne')




def get_bug_num(cutoff, X, y, mask, xl, xu, p=0, c=0.15, th=0.5):

    if cutoff == 0:
        return 0, 0, 0
    p = p
    c = c
    th = int(len(mask)*th)

    def process_specific_bug(bug_ind):
        chosen_bugs = y == bug_ind
        specific_bugs = X[chosen_bugs]
        unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(specific_bugs, mask, xl, xu, p, c, th)

        return len(unique_specific_bugs)

    unique_collision_num = process_specific_bug(1)
    unique_offroad_num = process_specific_bug(2)
    unique_wronglane_num = process_specific_bug(3)

    return unique_collision_num, unique_offroad_num, unique_wronglane_num


def unique_bug_num_seq_partial_objectives(path_list):


    all_X_list = []
    all_y_list = []

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)

        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_X = d['X'][inds]
        all_y = d['y'][inds]
        objectives = objectives[inds]

        all_X_list.append(all_X)
        all_y_list.append(all_y)

    all_X = np.concatenate(all_X_list)
    all_y = np.concatenate(all_y_list)

    collision_num, offroad_num, wronglane_num = get_bug_num(700, all_X, all_y, mask, xl, xu)
    print(collision_num, offroad_num, wronglane_num)


def analyze_objectives(path_list, filename='objectives_bug_num_over_simulations', scene_name=''):




    cutoffs = [100*i for i in range(0, 8)]
    data_list = []
    labels = []

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)
        labels.append(label)

        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_X = d['X'][inds]
        all_y = d['y'][inds]
        objectives = objectives[inds]


        data = []
        for cutoff in cutoffs:
            X = all_X[:cutoff]
            y = all_y[:cutoff]
            collision_num, offroad_num, wronglane_num = get_bug_num(cutoff, X, y, mask, xl, xu)

            if cutoff == 1400:
                print(collision_num, offroad_num, wronglane_num)

            speed = np.mean(objectives[:cutoff, 0])
            min_d = np.mean(objectives[:cutoff, 1])
            offroad = np.mean(objectives[:cutoff, 2])
            wronglane = np.mean(objectives[:cutoff, 3])
            dev = np.mean(objectives[:cutoff, 4])

            bug_num = collision_num+offroad_num+wronglane_num
            out_of_road_num = offroad_num+wronglane_num
            data.append(np.array([bug_num, collision_num, offroad_num, wronglane_num, out_of_road_num, speed, min_d, offroad, wronglane, dev]))

        data = np.stack(data)
        data_list.append(data)




    fig = plt.figure(figsize=(12.5, 5))

    # fig = plt.figure(figsize=(15, 9))
    # plt.suptitle("values over simulations", fontsize=14)


    # info = [(1, 0, 'Bug Numbers'), (6, 1, 'Collision Numbers'), (7, 2, 'Offroad Numbers'), (8, 3, 'Wronglane Numbers'), (9, 4, 'Out-of-road Numbers'), (11, 5, 'Collision Speed'), (12, 6, 'Min object distance'), (13, 7, 'Offroad Directed Distance'), (14, 8, 'Wronglane Directed Distance'), (15, 9, 'Max Deviation')]

    info = [(1, 1, '# unique collision'), (2, 4, '# unique out-of-road')]

    for loc, ind, ylabel in info:
        ax = fig.add_subplot(1, 2, loc)
        for i in range(len(data_list)):
            label = labels[i]
            y = data_list[i][:, ind]
            ax.plot(cutoffs, y, label=label, linewidth=2, marker='o', markersize=10)
        if loc == 1:
            ax.legend(loc=2, prop={'size': 26}, fancybox=True, framealpha=0.2)

            # import pylab
            # fig_p = pylab.figure()
            # figlegend = pylab.figure(figsize=(3,2))
            # ax = fig_p.add_subplot(111)
            # lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10))
            # figlegend.legend(lines, ('collision-', 'two'), 'center')
            # fig.show()
            # figlegend.show()
            # figlegend.savefig('legend.png')

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.set_xlabel("# simulations", fontsize=26)
        ax.set_ylabel(ylabel, fontsize=26)

    fig.suptitle(scene_name, fontsize=38)

    fig.tight_layout()



    plt.savefig(filename)


def ablate_thresholds(path_list, thresholds_list, cutoff):

    p = 0

    xl = None
    xu = None
    mask = None

    for c, th in thresholds_list:
        print('(', c, th, ')')
        for i, (label, pth) in enumerate(path_list):
            print(label)
            d = np.load(pth, allow_pickle=True)

            if i == 0:
                xl = d['xl']
                xu = d['xu']
                mask = d['mask']
            objectives = np.stack(d['objectives'])
            df_objectives = np.array(default_objectives)

            eps = 1e-7
            diff = np.sum(objectives - df_objectives, axis=1)


            inds = np.abs(diff) > eps

            all_X = d['X'][inds]
            all_y = d['y'][inds]
            objectives = objectives[inds]


            X = all_X[:cutoff]
            y = all_y[:cutoff]
            collision_num, offroad_num, wronglane_num = get_bug_num(cutoff, X, y, mask, xl, xu, p=p, c=c, th=th)

            print(collision_num+offroad_num+wronglane_num, collision_num, offroad_num, wronglane_num)


def check_unique_bug_num(folder, path1, path2):
    d = np.load(folder+'/'+path1, allow_pickle=True)
    xl = d['xl']
    xu = d['xu']
    mask = d['mask']

    d = np.load(folder+'/'+path2, allow_pickle=True)
    all_X = d['X']
    all_y = d['y']
    cutoffs = [100*i for i in range(0, 15)]


    def subroutine(cutoff):
        if cutoff == 0:
            return 0, []
        X = all_X[:cutoff]
        y = all_y[:cutoff]

        bugs = X[y>0]


        p = 0
        c = 0.15
        th = int(len(mask)*0.5)

        filtered_bugs, inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)
        print(cutoff, len(filtered_bugs), len(bugs))
        return len(filtered_bugs), inds


    num_of_unique_bugs = []
    for cutoff in cutoffs:
        num, inds = subroutine(cutoff)
        num_of_unique_bugs.append(num)
    print(inds)
    # print(bug_counters)
    # counter_inds = np.array(bug_counters)[inds] - 1
    # print(all_X[counter_inds[-2]])
    # print(all_X[counter_inds[-1]])

    plt.plot(cutoffs, num_of_unique_bugs, marker='o', markersize=10)
    plt.xlabel('# simulations')
    plt.ylabel('# unique violations')
    plt.savefig('num_of_unique_bugs')




def draw_hv_and_gd(path_list):
    from pymoo.factory import get_performance_indicator
    def is_pareto_efficient_dumb(costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
        return is_efficient

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)
        X = d['X']
        y = d['y']
        objectives = d['objectives'][:, :5] * np.array([-1, 1, 1, 1, -1])

        pareto_set = objectives[is_pareto_efficient_dumb(objectives)]
        # print(label, np.sum(is_pareto_efficient_dumb(objectives)))




        gd = get_performance_indicator("gd", pareto_set)
        hv = get_performance_indicator("hv", ref_point=np.array([0.01, 7.01, 7.01, 7.01, 0.01]))

        print(label)
        for j in range(16):
            cur_objectives = objectives[:(j+1)*100]
            print(j)
            print("GD", gd.calc(cur_objectives))
            print("hv", hv.calc(cur_objectives))




def calculate_pairwise_dist(path_list):
    xl = None
    xu = None
    mask = None
    for i, (label, pth) in enumerate(path_list):
        print(label)
        d = np.load(pth, allow_pickle=True)
        if i == 0:
            xl = d['xl']
            xu = d['xu']
            mask = d['mask']
            print(len(mask))

        p = 0
        c = 0.15
        th = int(len(mask)*0.5)

        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_y = d['y'][inds][:1500]
        all_X = d['X'][inds][:1500]


        # all_X, inds = get_distinct_data_points(all_X, mask, xl, xu, p, c, th)
        # all_y = all_y[inds]

        int_inds = mask == 'int'
        real_inds = mask == 'real'
        eps = 1e-8




        def pair_dist(x_1, x_2):
            int_diff_raw = np.abs(x_1[int_inds] - x_2[int_inds])
            int_diff = np.ones(int_diff_raw.shape) * (int_diff_raw > eps)

            real_diff_raw = np.abs(x_1[real_inds] - x_2[real_inds]) / (np.abs(xu[real_inds] - xl[real_inds]) + eps)

            real_diff = np.ones(real_diff_raw.shape) * (real_diff_raw > c)

            diff = np.concatenate([int_diff, real_diff])

            diff_norm = np.linalg.norm(diff, p)
            # print(diff, diff_norm)
            return diff_norm



        dist_list = []
        for i in range(len(all_X)-1):
            for j in range(i+1, len(all_X)):
                if check_bug(objectives[i]) > 0 and check_bug(objectives[j]) > 0:
                    diff = pair_dist(all_X[i], all_X[j])
                    if diff:
                        dist_list.append(diff)


        dist = np.array(dist_list) / len(mask)
        print(np.mean(dist), np.std(dist))




def draw_unique_bug_num_over_simulations(path_list, filename='num_of_unique_bugs', scene_name='', legend=True, range_upper_bound=16):
    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    line_style = ['-', '-', '-', '-']
    from ga_fuzzing import default_objectives
    xl = None
    xu = None
    mask = None
    for i, (label, pth) in enumerate(path_list):
        print(label)
        d = np.load(pth, allow_pickle=True)
        if i == 0:
            xl = d['xl']
            xu = d['xu']
            mask = d['mask']
            print(len(mask))
        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)
        # print(objectives.shape, df_objectives.shape)
        # print(objectives == df_objectives)
        # print(objectives[500:600])

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)
        # print(diff[500:600])
        # print((diff>eps)[500:600])

        inds = np.abs(diff) > eps

        all_X = d['X'][inds]
        all_y = d['y'][inds]
        cutoffs = [100*i for i in range(0, range_upper_bound)]



        def subroutine(cutoff):
            if cutoff == 0:
                return 0, []
            X = all_X[:cutoff]
            y = all_y[:cutoff]

            bugs = X[y>0]


            p = 0
            c = 0.15
            th = int(len(mask)*0.5)

            filtered_bugs, inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)
            print(cutoff, len(filtered_bugs), len(bugs))
            return len(filtered_bugs), inds


        num_of_unique_bugs = []
        for cutoff in cutoffs:
            num, inds = subroutine(cutoff)
            num_of_unique_bugs.append(num)

        print(inds)
        # print(bug_counters)
        # counter_inds = np.array(bug_counters)[inds] - 1
        # print(all_X[counter_inds[-2]])
        # print(all_X[counter_inds[-1]])

        axes.plot(cutoffs, num_of_unique_bugs, label=label, linewidth=2, linestyle=line_style[i], marker='o', markersize=10)
    axes.set_title(scene_name, fontsize=26)
    if legend:
        axes.legend(loc=2, prop={'size': 26}, fancybox=True, framealpha=0.2)
    axes.set_xlabel('# simulations', fontsize=26)
    axes.set_ylabel('# unique violations', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.tight_layout()
    fig.savefig(filename)

if __name__ == '__main__':
    


    # check_unique_bug_num('data_for_analysis/2020_08_15_17_21_03_12_100_leading_car_all_objective', 'data_for_analysis/2020_08_15_17_21_03_12_100_leading_car_all_objective/Town05_Scenario12_right_0_leading_car_braking_12_100_all_objectives_2020_08_16_00_53_08.npz')



    # check_unique_bug_num('data_for_analysis/', 'new_town05_right_50_10/2020_08_22_02_59_38_nsga2/bugs/town05_right_0_leading_car_braking_town05_lbc_15_100.npz', 'new_town05_right_50_10/2020_08_22_02_59_38_nsga2/bugs/town05_right_0_leading_car_braking_town05_lbc_15_100.npz')
    #
    # check_unique_bug_num('data_for_analysis/', 'new_town05_right_50_10/2020_08_22_02_59_38_nsga2/bugs/town05_right_0_leading_car_braking_town05_lbc_15_100.npz', 'new_town05_right_50_10/2020_08_22_03_00_00_nsga2_dt/town05_right_0_leading_car_braking_town05_lbc_5_100_3_2020_08_22_03_00_00.npz')


    # check_unique_bug_num('data_for_analysis/', 'new_town07_front_50_10/2020_08_23_03_24_33_nsga2-un/bugs/nsga2-un_town07_front_0_low_traffic_lbc_15_100.npz', 'new_town07_front_50_10/2020_08_23_03_24_33_nsga2-un/bugs/nsga2-un_town07_front_0_low_traffic_lbc_15_100.npz')


    # check_unique_bug_num('data_for_analysis/', 'new_nsga2-un_town07_front_50_10/2020_08_23_03_24_29_random/bugs/random_town07_front_0_low_traffic_lbc_15_100.npz', 'new_nsga2-un_town07_front_50_10/2020_08_23_03_24_29_random/bugs/random_town07_front_0_low_traffic_lbc_15_100.npz')






    # town07
    town07_path_list = [('Random', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_03_24_29_random_50_10/bugs/random_town07_front_0_low_traffic_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_11_58_10_nsga2_50_15/bugs/nsga2_town07_front_0_low_traffic_lbc_15_100.npz'), ('NSGA2-UN', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_11_58_24_nsga2-un_50_15/bugs/nsga2-un_town07_front_0_low_traffic_lbc_15_100.npz'), ('NSGA2-DT', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_26_21_31_37_nsga2-dt_50_15_full/nsga2-dt_town07_front_0_low_traffic_lbc_5_100_15_2020_08_26_21_31_37.npz')]


    town01_path_list = [('Random', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_23_20_59_53_random/bugs/random_town01_left_0_default_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_24_11_37_12_nsga2/bugs/nsga2_town01_left_0_default_lbc_15_100.npz'), ('NSGA2-UN', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_23_20_59_46_nsga2-un/bugs/nsga2-un_town01_left_0_default_lbc_15_100.npz'), ('NSGA2-DT', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_26_17_56_47_nsga2-dt_full/nsga2-dt_town01_left_0_default_lbc_5_100_15_2020_08_26_17_56_47.npz')]



    town05_front_path_list = [('Random', 'data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_24_01_35_09_random/bugs/random_town05_front_0_change_lane_town05_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_24_11_37_14_nsga2/bugs/nsga2_town05_front_0_change_lane_town05_lbc_15_100.npz'), ('NSGA2-UN', 'data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_23_21_42_33_nsga2-un/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_15_100.npz'), ('NSGA2-DT', 'data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_26_23_41_18_nsga2-dt_full/nsga2-dt_town05_front_0_change_lane_town05_lbc_5_100_15_2020_08_26_23_41_18.npz')]


    pids_path_list = [('pid-1', 'data_for_analysis/other_controllers/2020_08_26_11_39_16_autopilot_pid1/bugs/nsga2-un_town05_right_0_leading_car_braking_town05_auto_pilot_30_100.npz'), ('pid-2', 'data_for_analysis/other_controllers/2020_08_26_11_39_22_pid_pid2/bugs/nsga2-un_town05_right_0_leading_car_braking_town05_pid_agent_30_100.npz'), ('lbc', 'data_for_analysis/new_nsga2-un_town05_right_50_15/2020_08_26_18_43_03_nsga2-un/bugs/nsga2-un_town05_right_0_leading_car_braking_town05_lbc_30_100.npz')]

    # objectives_path_list1 = [('all objectives', 'data_for_analysis/new_nsga2-un_town05_right_50_15/2020_08_26_18_43_03_nsga2-un/bugs/nsga2-un_town05_right_0_leading_car_braking_town05_lbc_30_100.npz'), ('collision-only objectives', 'data_for_analysis/objectives_analysis/town_05_right/2020_08_26_11_08_05_partial_collision/bugs/nsga2-un_town05_right_0_leading_car_braking_town05_lbc_15_100.npz'), ('out-of-road-only objectives', 'data_for_analysis/objectives_analysis/town_05_right/2020_08_26_18_02_09_partial_out_of_road/bugs/nsga2-un_town05_right_0_leading_car_braking_town05_lbc_15_100.npz')]
    #
    # objectives_path_list2 = [('all objectives', 'data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_23_21_42_33_nsga2-un/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_15_100.npz'), ('collision-only objectives', 'data_for_analysis/objectives_analysis/town_05_front/2020_08_27_12_34_59_collision/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_30_100.npz'), ('out-of-road-only objectives', 'data_for_analysis/objectives_analysis/town_05_front/2020_08_27_12_34_50_out_of_road/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_30_100.npz')]



    objectives_path_list1 = [('collision-only\nobjectives', 'data_for_analysis/objectives_analysis/town_05_right/2020_08_26_11_08_05_partial_collision/bugs/nsga2-un_town05_right_0_leading_car_braking_town05_lbc_15_100.npz'), ('out-of-road-only\nobjectives', 'data_for_analysis/objectives_analysis/town_05_right/2020_08_26_18_02_09_partial_out_of_road/bugs/nsga2-un_town05_right_0_leading_car_braking_town05_lbc_15_100.npz')]

    objectives_path_list2 = [('collision-only\nobjectives', 'data_for_analysis/objectives_analysis/town_05_front/2020_08_27_12_34_59_collision/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_30_100.npz'), ('out-of-road-only\nobjectives', 'data_for_analysis/objectives_analysis/town_05_front/2020_08_27_12_34_50_out_of_road/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_30_100.npz')]


    sensitivity_path_list = [('Random', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_23_20_59_53_random/bugs/random_town01_left_0_default_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_24_11_37_12_nsga2/bugs/nsga2_town01_left_0_default_lbc_15_100.npz'), ('NSGA2-UN-0.15-0.5', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_23_20_59_46_nsga2-un/bugs/nsga2-un_town01_left_0_default_lbc_15_100.npz'), ('NSGA2-UN-0.075-0.5', 'data_for_analysis/sensitivity/2020_08_27_16_37_53_0.075_0.5/bugs/nsga2-un_town01_left_0_default_lbc_6_100.npz'), ('NSGA2-UN-0.075-0.25', 'data_for_analysis/sensitivity/2020_08_27_18_23_35_0.075_0.25/bugs/nsga2-un_town01_left_0_default_lbc_6_100.npz'), ('NSGA2-DT', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_26_17_56_47_nsga2-dt_full/nsga2-dt_town01_left_0_default_lbc_5_100_15_2020_08_26_17_56_47.npz')]


    sensitivity_path_list2 = [('Random', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_23_20_59_53_random/bugs/random_town01_left_0_default_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_24_11_37_12_nsga2/bugs/nsga2_town01_left_0_default_lbc_15_100.npz'), ('NSGA2-UN-0.15-0.5', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_23_20_59_46_nsga2-un/bugs/nsga2-un_town01_left_0_default_lbc_15_100.npz'), ('NSGA2-DT', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_26_17_56_47_nsga2-dt_full/nsga2-dt_town01_left_0_default_lbc_5_100_15_2020_08_26_17_56_47.npz')]



    # high_dim_path_list = [('random', 'data_for_analysis/high_dim_scene/2020_08_28_00_11_20_random/bugs/random_town04_front_0_pedestrians_cross_street_town04_lbc_30_100.npz'), ('NSGA2', 'data_for_analysis/high_dim_scene/2020_08_28_00_10_58_nsag2/bugs/nsga2_town04_front_0_pedestrians_cross_street_town04_lbc_30_100.npz'), ('NSGA2-UN', 'data_for_analysis/high_dim_scene/2020_08_27_11_47_58_nsga2-un/bugs/nsga2-un_town04_front_0_pedestrians_cross_street_town04_lbc_30_100.npz'), ('NSGA2-DT', 'data_for_analysis/high_dim_scene/2020_08_28_01_00_30_nsga2-dt/nsga2-dt_town04_front_0_pedestrians_cross_street_town04_lbc_5_100_15_2020_08_28_01_00_30.npz')]

    high_dim_path_list = [('NSGA2-UN', 'data_for_analysis/high_dim_scene/2020_08_27_11_47_58_nsga2-un/bugs/nsga2-un_town04_front_0_pedestrians_cross_street_town04_lbc_30_100.npz')]

    thresholds_list = [(0, 0), (0.075, 0.25), (0.075, 0.5), (0.075, 0.75), (0.15, 0.25), (0.15, 0.5), (0.15, 0.75), (0.225, 0.25), (0.225, 0.5), (0.225, 0.75)]

    # ablate_thresholds(sensitivity_path_list, thresholds_list, 300)
    # ablate_thresholds(town05_front_path_list, thresholds_list, 300)
    # draw other controllers
    # draw_unique_bug_num_over_simulations(pids_path_list, filename='num_of_unique_bugs_other_controllers')
    #
    #
    # draw_unique_bug_num_over_simulations(high_dim_path_list, filename='num_of_unique_bugs_town04_front', scene_name='', legend=False, range_upper_bound=16)
    #
    # draw_unique_bug_num_over_simulations(town01_path_list, filename='num_of_unique_bugs_town01_left', scene_name='turning left non-signalized', legend=True)
    #
    # draw_unique_bug_num_over_simulations(town05_front_path_list, filename='num_of_unique_bugs_town05_front', scene_name='crossing non-signalized', legend=False)
    #
    # draw_unique_bug_num_over_simulations(town07_path_list, filename='num_of_unique_bugs_town07_front', scene_name='changing lane', legend=False)


    # analyze different objectives
    analyze_objectives(objectives_path_list1, filename='objectives_bug_num_over_simulations_town05_right', scene_name='leading car slows down / stops')
    analyze_objectives(objectives_path_list2, filename='objectives_bug_num_over_simulations_town05_front', scene_name='changing lane')

    # unique_bug_num_seq_partial_objectives(objectives_path_list2)

    # calculate_pairwise_dist(town05_front_path_list)
    # calculate_pairwise_dist(town07_path_list)
    # calculate_pairwise_dist(town01_path_list)

    # draw_hv_and_gd(town05_front_path_list)
    # draw_hv_and_gd(town07_path_list)
    # draw_hv_and_gd(town01_path_list)





    # apply_tsne('data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_23_21_42_33_nsga2-un/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_15_100.npz', 15, 100)

import pandas as pd
import numpy as np
import os


# ['train', 'test', 'learn']
mode = 'test'

weather_indexes = [15]
routes = [i for i in range(76)]
routes.remove(13)
model_name = 'SAE'
data_dir = 'collected_data'


if mode == 'test':


    infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane']


    behaviors_list = []
    features_list = []
    behaviors_names = []
    misbehavior_names = []
    settings_length = []

    for weather_id in weather_indexes:
        for route in routes:
            route_str = str(route)
            if route < 10:
                route_str = '0'+route_str

            data_df = pd.read_csv(os.path.join(data_dir, 'route_'+route_str+'_'+str(weather_id), 'driving_log.csv'))
            behaviors_list.append(data_df['behaviors'])
            behaviors_names.extend(data_df['behaviors_names'])
            features_list.append(np.load(os.path.join(data_dir, 'route_'+route_str+'_'+str(weather_id), model_name+'_simple-autoencoder-model-collected_data'+'_features.npy')))
            misbehavior_names.extend(data_df['Misbehavior'])

            settings_length.append(len(data_df['behaviors']))



    behaviors = np.concatenate(behaviors_list, axis=0)
    features = np.concatenate(features_list, axis=0)
    settings_length = np.array(settings_length)
    print(behaviors.shape, features.shape, len(behaviors_names), len(misbehavior_names))
    np.savez(data_dir+'/'+model_name+'_'+'_'.join([str(w_id) for w_id in weather_indexes]), behaviors=behaviors, features=features, behaviors_names=behaviors_names, misbehavior_names=misbehavior_names, settings_length=settings_length)

elif mode == 'train':
    features = np.load('collected_data/train_simple-autoencoder-model-collected_data_features.npy')
    print(features.shape)

    behaviors_list = []

    behaviors_names = []
    misbehavior_names = []

    for weather_id in weather_indexes:
        for route in routes:
            route_str = str(route)
            if route < 10:
                route_str = '0'+route_str

            data_df = pd.read_csv(os.path.join(data_dir, 'route_'+route_str+'_'+str(weather_id), 'driving_log.csv'))
            behaviors_list.append(data_df['behaviors'])
            behaviors_names.extend(data_df['behaviors_names'])
            misbehavior_names.extend(data_df['Misbehavior'])

    behaviors = np.concatenate(behaviors_list, axis=0)
    print(behaviors.shape, features.shape, len(behaviors_names), len(misbehavior_names))
    np.savez(data_dir+'/'+'train_'+model_name+'_'+'_'.join([str(w_id) for w_id in weather_indexes]), behaviors=behaviors, features=features, behaviors_names=behaviors_names, misbehavior_names=misbehavior_names)

elif mode == 'learn':
    # ['ignore_end_of_stream', 'normal', 'reaction', 'None', 'gap', 'healing', 'anomaly', 'misbehavior']
    d = np.load('collected_data/SAE_15.npz')
    inputs = d['features']
    settings_length = d['settings_length']


    labels_set = ['normal', 'anomaly', 'ignore_end_of_stream', 'gap', 'reaction', 'misbehavior', 'healing', 'None']
    mapping = {labels_set[i]:i for i in range(len(labels_set))}
    labels = np.array([mapping[b] for b in d['behaviors_names']])
    n = labels.shape[0]



    inputs_list = []
    labels_list = []
    s = 0
    for l in settings_length:
        inputs_list.append(inputs[s:s+l])
        labels_list.append(labels[s:s+l])
        s += l

    num_of_settings = len(settings_length)
    num_of_settings_train = int(num_of_settings * 0.8)
    print(num_of_settings, num_of_settings_train)
    import random
    random.Random(0).shuffle(inputs_list)
    random.Random(0).shuffle(labels_list)

    train_x = np.concatenate(inputs_list[:num_of_settings_train])
    test_x = np.concatenate(inputs_list[num_of_settings_train:])
    train_y = np.concatenate(labels_list[:num_of_settings_train])
    test_y = np.concatenate(labels_list[num_of_settings_train:])



    inds = np.where((train_y==0) | (train_y==4))
    train_x = train_x[inds]
    train_y = train_y[inds]
    inds = np.where((test_y==0) | (test_y==4))
    test_x = test_x[inds]
    test_y = test_y[inds]

    train_y[train_y!=0] = 1
    test_y[test_y!=0] = 1

    print('train', 'positive', np.sum(train_y==1), 'negative', np.sum(train_y==0))
    print('test', 'positive', np.sum(test_y==1), 'negative', np.sum(test_y==0))


    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    print(train_x.shape, train_y.shape)

    # clf = LogisticRegression().fit(train_x, train_y)
    clf = KNeighborsClassifier(n_neighbors=5).fit(train_x, train_y)

    y_pred = clf.predict(test_x)

    print(np.mean(y_pred))
    print(np.mean(train_y), np.mean(test_y))

    # for i in range(2):
    #     score_i = np.mean((y_pred==test_y) * (test_y == i))
    #     print(score_i)

    total = test_y.shape[0]
    positive = np.sum(y_pred == 1)
    true = np.sum(test_y == 1)

    tp = np.sum((y_pred == 1) * (test_y == 1))
    fp = np.sum((y_pred == 1) * (test_y == 0))
    tn = np.sum((y_pred == 0) * (test_y == 0))
    fn = np.sum((y_pred == 0) * (test_y == 1))

    prec = tp/(tp+fp)
    recall = tp/(tp+fn)

    r_prec = true/total
    r_recall = positive**2/(total*true)

    print(prec, recall, tp, fp, tn, fn)
    print(r_prec, r_recall)

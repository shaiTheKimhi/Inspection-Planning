import argparse
from MapEnvironment import MapEnvironment
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
import time
import os
import pickle

os.system('rm -f *.gif')

task = 'mp'
map = 'map_mp.json'
planning_env = MapEnvironment(json_file=map, task=task)
mp_results = {}
for mode in ['E1', 'E2']:
    mp_results[mode] = {}
    for goal_prob in [0.05, 0.2]:
        mp_results[mode][goal_prob] = {'cost': [], 'time': []}
        for i in range(10):
            print('mode: ' + mode + '    goal_prob: ' + str(goal_prob) + '    iter: ' + str(i+1) + '/' + str(10))
            planner = RRTMotionPlanner(planning_env=planning_env, ext_mode='E1', goal_prob=0.2)
            start_time = time.time()
            plan = planner.plan()
            mp_results[mode][goal_prob]['time'].append(time.time() - start_time)
            mp_results[mode][goal_prob]['cost'].append(planner.compute_cost(plan))
            planner.planning_env.visualize_plan(plan)
with open('mp.pickle', 'bw') as f:
    pickle.dump(mp_results, f)

task = 'ip'
map = 'map_ip.json'
planning_env = MapEnvironment(json_file=map, task=task)
ip_results = {}
for mode in ['E1', 'E2']:
    ip_results[mode] = {}
    for goal_prob in [0.05, 0.2]:
        ip_results[mode][goal_prob] = {'cost': [], 'time': []}
        for coverage in [0.75, 0.5]:
            ip_results[mode][goal_prob] = {'cost': [], 'time': []}
            for i in range(10):
                print('mode: ' + mode + '    goal_prob: ' + str(goal_prob) + '    coverage: ' + str(coverage) + '    iter: ' + str(i + 1) + '/' + str(10))
                planner = RRTInspectionPlanner(planning_env=planning_env, ext_mode=mode, goal_prob=goal_prob, coverage=coverage)
                start_time = time.time()
                plan = planner.plan()
                ip_results[mode][goal_prob]['time'].append(time.time() - start_time)
                ip_results[mode][goal_prob]['cost'].append(planner.compute_cost(plan))
                planner.planning_env.visualize_plan(plan)

with open('ip.pickle', 'bw') as f:
    pickle.dump(ip_results, f)

print('\n\n\n')
print('----------------')
print('MOTION PLANNING:')
print('----------------')
for mode in ['E1', 'E2']:
    for goal_prob in [0.05, 0.2]:
        print('mode: ' + mode + '    goal_prob: ' + str(goal_prob) + '    avg_time: ' + str(np.mean(mp_results[mode][goal_prob]['time'])) + '    avg_cost: ' + str(np.mean(mp_results[mode][goal_prob]['cost'])))

print('\n\n')
print('----------------')
print('INSPECTION PLANNING:')
print('----------------')
for mode in ['E1', 'E2']:
    for goal_prob in [0.05, 0.2]:
        for coverage in [0.5, 0.75]:
            print('mode: ' + mode + '    goal_prob: ' + str(goal_prob) + '    coverage: ' + str(coverage) + '    avg_time: ' + str(np.mean(mp_results[mode][goal_prob]['time'])) + '    avg_cost: ' + str(np.mean(mp_results[mode][goal_prob]['cost'])))


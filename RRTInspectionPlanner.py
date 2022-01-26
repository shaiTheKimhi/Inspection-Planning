import numpy as np
from RRTTree import RRTTree
import time

class RRTInspectionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env, task="ip")

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # Initialize an empty plan.
        plan = []

        # TODO: Task 2.4

        # np.random.seed(1234)
        display = False
        n = 200000  # num iterations
        if self.ext_mode == 'E1':
            self.step_size = -1
        elif self.ext_mode == 'E2':
            self.step_size = 10. * np.pi / 180.
        else:
            return []


        # Start with adding the start configuration to the tree.
        self.tree.add_vertex(self.planning_env.start, inspected_points=self.planning_env.get_inspected_points(self.planning_env.start))
        x_root_id = self.tree.get_idx_for_config(self.planning_env.start)
        x_new_id = x_root_id
        x_new = None
        goal_points = {}
        explored_points = []
        max_coverage = 0.
        max_coverage_tree = RRTTree(self.planning_env, task="ip")
        max_coverage_tree.add_vertex(self.planning_env.start, inspected_points=self.planning_env.get_inspected_points(self.planning_env.start))
        for _ in range(n):
            # sampling
            if np.random.uniform() < self.goal_prob and len(goal_points) > 0:
                # mode = 'GOAL'
                x_rand = np.random.rand(*self.planning_env.start.shape) * 2. * np.pi - np.pi
                rand_inspected_points = self.planning_env.get_inspected_points(x_rand)
                if len(rand_inspected_points) > 0:
                    for poi in rand_inspected_points:
                        if str(poi) in goal_points.keys():
                            if str(poi) not in explored_points:
                                goal_points[str(poi)].append(x_rand)
                        else:
                            goal_points[str(poi)] = [x_rand]

                # nearest neighbor
                _, x_near = max_coverage_tree.get_nearest_config(x_rand)
                x_near_id = self.tree.get_idx_for_config(x_near)
            else:
                # mode = 'RAND'
                x_rand = np.random.rand(*self.planning_env.start.shape) * 2. * np.pi - np.pi
                rand_inspected_points = self.planning_env.get_inspected_points(x_rand)
                if len(rand_inspected_points) > 0:
                    for poi in rand_inspected_points:
                        if str(poi) in goal_points.keys():
                            if str(poi) not in explored_points:
                                goal_points[str(poi)].append(x_rand)
                        else:
                            goal_points[str(poi)] = [x_rand]

                # nearest neighbor
                x_near_id, x_near = self.tree.get_nearest_config(x_rand)

            # extend
            x_new = self.extend(x_near, x_rand)
            if x_new is None:
                continue

            new_inspected_points = self.planning_env.get_inspected_points(x_new)

            # collision detection
            if self.planning_env.config_validity_checker(x_new) and \
                    self.planning_env.edge_validity_checker(x_near, x_new):

                curr_union_pois = self.planning_env.compute_union_of_points(new_inspected_points, self.tree.vertices[x_near_id].inspected_points)
                x_new_id = self.tree.add_vertex(x_new, inspected_points=curr_union_pois)

                curr_coverage = self.planning_env.compute_coverage(self.tree.vertices[x_new_id].inspected_points)
                self.tree.add_edge(x_near_id, x_new_id, self.planning_env.robot.compute_distance(x_new, x_near))
                if curr_coverage >= self.coverage:
                    break
                elif curr_coverage > max_coverage:
                    max_coverage_tree = RRTTree(self.planning_env, task="ip")
                    max_coverage_tree.add_vertex(x_new, inspected_points=curr_union_pois)
                    max_coverage = curr_coverage
                    #print(max_coverage)
                elif curr_coverage == max_coverage:
                    max_coverage_tree.add_vertex(x_new, inspected_points=curr_union_pois)

        # build plan
        ratio = self.planning_env.compute_coverage(self.tree.vertices[x_new_id].inspected_points)
        eid = x_new_id if x_new is not None and ratio >= self.coverage else x_root_id
        while True:
            plan.append(self.tree.vertices[eid].config)
            if eid == x_root_id:
                break
            eid = self.tree.edges[eid]
        plan = np.array(plan)[::-1]

        if display:
            self.plot_plan(plan, dt=0.05)
            import pdb; pdb.set_trace()

        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def plot_plan(self, plan, dt=0.05):
        import matplotlib.pyplot as plt
        plt.close('all')
        self.planning_env.create_map_visualization()
        self.planning_env.visualize_obstacles(plt.gca())
        plt.scatter(self.planning_env.inspection_points[:, 0], self.planning_env.inspection_points[:, 1], c='r')
        mid_inspected_points_scatter = plt.scatter([], [], c='b')
        inspected_points_scatter = plt.scatter([], [], c='g')
        mid_inspected_points = None
        inspected_points = None
        points = np.concatenate([[[0., 0.]], self.planning_env.robot.compute_forward_kinematics(plan[0])])
        lp = plt.plot(points[:, 0], points[:, 1], linewidth=3, )[0]
        ls = plt.scatter(points[:, 0], points[:, 1], c='m')

        ee_angle = self.planning_env.robot.compute_ee_angle(given_config=plan[0])
        sensor_dist = self.planning_env.robot.vis_dist
        sensor_angles = np.linspace(ee_angle - self.planning_env.robot.ee_fov / 2.,
                                    ee_angle + self.planning_env.robot.ee_fov / 2., 100)
        sensor_points = points[-1] + np.concatenate(
            [[[0., 0.]], sensor_dist * np.array([np.cos(sensor_angles), np.sin(sensor_angles)]).T])
        sensor = plt.fill(sensor_points[:, 0], sensor_points[:, 1], c='y', alpha=0.3)[0]

        for i in range(len(plan) - 1):
            config1 = plan[i]
            config2 = plan[i + 1]
            interpolation_steps = int(np.linalg.norm(config2 - config1) // 0.05)
            interpolated_configs = np.linspace(start=config1, stop=config2, num=interpolation_steps)
            configs_positions = np.apply_along_axis(self.planning_env.robot.compute_forward_kinematics, 1,
                                                    interpolated_configs)
            lines = np.concatenate(
                [np.zeros((configs_positions.shape[0], 1, configs_positions.shape[2])), configs_positions], axis=1)
            for j in range(len(lines)):
                line = lines[j]
                config = interpolated_configs[j]
                curr_inspected_points = self.planning_env.get_inspected_points(config)
                if len(curr_inspected_points) > 0:
                    if mid_inspected_points is None:
                        mid_inspected_points = curr_inspected_points
                        mid_inspected_points_scatter.set_offsets(mid_inspected_points)
                    else:
                        mid_inspected_points = np.unique(
                            np.concatenate((mid_inspected_points, curr_inspected_points), axis=0), axis=0)
                        mid_inspected_points_scatter.set_offsets(mid_inspected_points)
                lp.set_data(line[:, 0], line[:, 1])
                ls.set_offsets(line)

                ee_angle = self.planning_env.robot.compute_ee_angle(given_config=config)
                sensor_angles = np.linspace(ee_angle - self.planning_env.robot.ee_fov / 2.,
                                            ee_angle + self.planning_env.robot.ee_fov / 2., 100)
                sensor_points = line[-1] + np.concatenate(
                    [[[0., 0.]], sensor_dist * np.array([np.cos(sensor_angles), np.sin(sensor_angles)]).T])
                sensor.set_xy(sensor_points)

                plt.pause(dt)
            curr_inspected_points = self.planning_env.get_inspected_points(config2)
            if len(curr_inspected_points) > 0:
                if inspected_points is None:
                    inspected_points = curr_inspected_points
                    inspected_points_scatter.set_offsets(inspected_points)
                else:
                    inspected_points = np.unique(np.concatenate((inspected_points, curr_inspected_points), axis=0), axis=0)
                    inspected_points_scatter.set_offsets(inspected_points)
            plt.pause(dt)


    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.4
        return np.sum(self.planning_env.robot.compute_distance(plan[1:], plan[:-1]))

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: Task 2.4

        angles_diff = ((rand_config - near_config + np.pi) % (2. * np.pi)) - np.pi
        abs_angles_diff = np.fabs(angles_diff)

        if np.all(abs_angles_diff == 0):
            return None
        if self.step_size < 0:
            return rand_config
        else:
            new_config = np.where(abs_angles_diff < self.step_size,
                                  rand_config,
                                  (
                                  (near_config + np.sign(angles_diff) * self.step_size + np.pi) % (2. * np.pi)) - np.pi)
            return new_config

    
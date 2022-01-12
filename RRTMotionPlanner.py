import numpy as np
from RRTTree import RRTTree
import time

class RRTMotionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.step_size = 5. * np.pi / 180.

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # Initialize an empty plan.
        plan = []

        # TODO: Task 2.3
        np.random.seed(1234)
        display = False
        n = 5000000  # num iterations
        if self.ext_mode == 'E1':
            self.step_size = -1
        elif self.ext_mode == 'E2':
            assert(self.step_size is not None and self.step_size > 0)
            self.step_size = self.step_size
        else:
            return []

        # Start with adding the start configuration to the tree.
        self.tree.add_vertex(self.planning_env.start)

        x_root_id = self.tree.get_idx_for_config(self.planning_env.start)
        x_new_id = x_root_id
        x_new = None
        import tqdm
        for _ in tqdm.tqdm(range(n)):
            # sampling
            if np.random.uniform() < self.goal_prob:
                x_rand = self.planning_env.goal
            else:
                x_rand = np.random.rand(*self.planning_env.goal.shape) * 2. * np.pi - np.pi

            # nearest neighbor
            x_near_id, x_near = self.tree.get_nearest_config(x_rand)

            # extend
            x_new = self.extend(x_near, x_rand)
            if x_new is None:
                continue

            # collision detection
            if self.planning_env.config_validity_checker(x_new) and \
               self.planning_env.edge_validity_checker(x_near, x_new):
                x_new_id = self.tree.add_vertex(x_new)
                self.tree.add_edge(x_near_id, x_new_id, self.planning_env.robot.compute_distance(x_new, x_near))
                if np.all(x_new == self.planning_env.goal):
                    break
        # build plan
        eid = x_new_id if x_new is not None and np.all(x_new == self.planning_env.goal) else x_root_id
        while True:
            plan.append(self.tree.vertices[eid].config)
            if eid == x_root_id:
                break
            eid = self.tree.edges[eid]
        plan = np.array(plan)[::-1]

        if display:
            import matplotlib.pyplot as plt
            self.planning_env.create_map_visualization()
            self.planning_env.visualize_obstacles(plt.gca())
            points = np.concatenate([[[0.,0.]], self.planning_env.robot.compute_forward_kinematics(plan[0])])
            lp = plt.plot(points[:,0], points[:,1],linewidth=3)[0]
            for i in range(len(plan) - 1):
                config1 = plan[i]
                config2 = plan[i + 1]
                interpolation_steps = int(np.linalg.norm(config2 - config1) // 0.05)
                interpolated_configs = np.linspace(start=config1, stop=config2, num=interpolation_steps)
                configs_positions = np.apply_along_axis(self.planning_env.robot.compute_forward_kinematics, 1, interpolated_configs)
                lines = np.concatenate([np.zeros((configs_positions.shape[0], 1, configs_positions.shape[2])), configs_positions], axis=1)
                for line in lines:
                    lp.set_data(line[:, 0], line[:, 1])
                    plt.pause(0.05)
            plt.close('all')

        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.3
        return np.sum(self.planning_env.robot.compute_distance(plan[1:], plan[:-1]))

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: Task 2.3
        angles_diff = ((rand_config - near_config + np.pi) % (2. * np.pi)) - np.pi
        abs_angles_diff = np.fabs(angles_diff)

        if np.all(abs_angles_diff == 0):
            return None
        if self.step_size < 0:
            return rand_config
        else:
            new_config = np.where(abs_angles_diff < self.step_size,
                                  rand_config,
                                  ((near_config + np.sign(angles_diff) * self.step_size + np.pi) % (2. * np.pi)) - np.pi)
            return new_config
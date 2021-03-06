import itertools
import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from shapely.geometry import Point, LineString

class Robot(object):
    
    def __init__(self):

        # define robot properties
        self.links = np.array([80.0,70.0,40.0,40.0]) #the length of each link
        self.dim = len(self.links)

        # Robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi/3

        # Visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        #Task 2.2
        # return np.linalg.norm(np.fabs(((next_config - prev_config + np.pi) % (2. * np.pi)) - np.pi))
        #Computes the distance between the edge inspector, TODO: check if to compare all links or just edges
        if len(prev_config.shape) > 1:
            p1 = np.apply_along_axis(self.compute_forward_kinematics, 1, prev_config)
            p2 = np.apply_along_axis(self.compute_forward_kinematics, 1, next_config)
            return np.linalg.norm(p1 - p2, axis=(-1, -2))
        else:
            p1 = self.compute_forward_kinematics(prev_config)
            p2 = self.compute_forward_kinematics(next_config)
        return np.linalg.norm(p1 - p2)

    def compute_forward_kinematics(self, given_config):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        # TODO: Task 2.2
        angles_rel_x = np.cumsum(given_config)
        vecs = np.array([self.links]).T * np.stack([np.cos(angles_rel_x), np.sin(angles_rel_x)]).T
        return np.cumsum(vecs, axis=0)


    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1,len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        if link_angle + given_angle > np.pi:
            return link_angle + given_angle - 2*np.pi
        elif link_angle + given_angle < -np.pi:
            return link_angle + given_angle + 2*np.pi
        else:
            return link_angle + given_angle
        
    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        # TODO: Task 2.2
        positions = [tuple(loc) for loc in robot_positions]
        return LineString(positions).is_simple

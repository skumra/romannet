#!/usr/bin/env python

import argparse
import concurrent.futures
import logging
import os
import re
import threading
import time

import cv2
import numpy as np
import tensorboardX
import torch
from scipy import ndimage

from robot import SimRobot
from trainer import Trainer
from utils import utils, viz
from utils.logger import Logger


class LearnManipulation:
    def __init__(self, args):
        # --------------- Setup options ---------------
        self.is_sim = args.is_sim
        sim_port = args.sim_port
        obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if self.is_sim else None
        num_obj = args.num_obj if self.is_sim else None
        # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        if self.is_sim:
            self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.8]])
        else:
            self.workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])
        self.heightmap_resolution = args.heightmap_resolution
        random_seed = args.random_seed
        force_cpu = args.force_cpu

        # ------------- Algorithm options -------------
        network = args.network
        num_rotations = args.num_rotations
        self.future_reward_discount = args.future_reward_discount
        self.explore_actions = args.explore_actions
        self.explore_type = args.explore_type
        self.explore_rate_decay = args.explore_rate_decay
        self.LAE_sigma = 0.33
        self.LAE_beta = 0.25
        self.experience_replay_disabled = args.experience_replay_disabled
        self.push_enabled = args.push_enabled
        self.place_enabled = args.place_enabled
        self.max_iter = args.max_iter
        self.reward_type = args.reward_type
        self.filter_type = args.filter_type
        self.place_reward_scale = args.place_reward_scale
        self.goal_stack_height = args.goal_stack_height

        # -------------- Testing options --------------
        self.is_testing = args.is_testing
        self.max_test_trials = args.max_test_trials
        test_preset_cases = args.test_preset_cases
        test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

        # ------ Pre-loading and logging options ------
        if args.logging_directory and not args.snapshot_file:
            logging_directory = os.path.abspath(args.logging_directory)
            self.snapshot_file = os.path.join(logging_directory, 'models/snapshot-backup.pth')
        elif args.snapshot_file:
            logging_directory = os.path.abspath(args.logging_directory)
            self.snapshot_file = os.path.abspath(args.snapshot_file)
        else:
            logging_directory = None
            self.snapshot_file = None

        self.save_visualizations = args.save_visualizations

        # Initialize pick-and-place system (camera and robot)
        if self.is_sim:
            self.robot = SimRobot(sim_port, obj_mesh_dir, num_obj, self.workspace_limits, self.is_testing,
                                  test_preset_cases, test_preset_file, self.place_enabled)
        else:
            raise NotImplementedError

        # Initialize data logger
        self.logger = Logger(logging_directory, args)
        self.logger.save_camera_info(self.robot.cam_intrinsics, self.robot.cam_pose, self.robot.cam_depth_scale)
        self.logger.save_heightmap_info(self.workspace_limits, self.heightmap_resolution)

        # Tensorboard
        self.tb = tensorboardX.SummaryWriter(logging_directory)

        # Initialize trainer
        self.trainer = Trainer(network, force_cpu, self.push_enabled, self.place_enabled, num_rotations)

        # Find last executed iteration of pre-loaded log, and load execution info and RL variables
        if self.logger.logging_directory_exists and not self.is_testing:
            self.trainer.preload(self.logger.transitions_directory)
            self.trainer.load_snapshot(self.snapshot_file)
        elif args.snapshot_file:
            self.trainer.load_snapshot(self.snapshot_file)

        # Set random seed
        np.random.seed(random_seed)

        # Initialize variables for heuristic bootstrapping and exploration probability
        self.no_change_count = [2, 2] if not self.is_testing else [0, 0]
        self.explore_prob = 0.5 if not self.is_testing else 0.0

        self.mission_complete = False
        self.execute_action = False
        self.shutdown_called = False

        self.prev_primitive_action = None
        self.prev_grasp_success = None
        self.prev_push_success = None
        self.prev_place_success = None
        self.prev_color_heightmap = None
        self.prev_depth_heightmap = None
        self.prev_best_pix_ind = None
        self.prev_stack_height = 0
        self.last_task_complete = 0

        self.push_predictions = None
        self.grasp_predictions = None
        self.place_predictions = None
        self.color_heightmap = None
        self.depth_heightmap = None
        self.primitive_action = None
        self.best_pix_ind = None
        self.predicted_value = None

    def policy(self):
        """
        Determine whether grasping or pushing or placing should be executed based on network predictions
        """
        best_push_conf = np.max(self.push_predictions)
        best_grasp_conf = np.max(self.grasp_predictions)
        best_place_conf = np.max(self.place_predictions)
        logging.info('Primitive confidence scores: %f (push), %f (grasp), %f (place)' % (
            best_push_conf, best_grasp_conf, best_place_conf))

        # Exploitation (do best action) vs exploration (do other action)
        if self.explore_actions and not self.is_testing:
            explore_actions = np.random.uniform() < self.explore_prob
            logging.info('Strategy: explore (exploration probability: %f)' % self.explore_prob)
        else:
            explore_actions = False

        self.trainer.is_exploit_log.append([0 if explore_actions else 1])
        self.logger.write_to_log('is-exploit', self.trainer.is_exploit_log)

        # Select action type
        self.primitive_action = 'grasp'
        if self.place_enabled and self.prev_primitive_action == 'grasp' and self.prev_grasp_success:
            self.primitive_action = 'place'
        elif self.push_enabled:
            if best_push_conf > best_grasp_conf:
                self.primitive_action = 'push'
            if explore_actions:
                self.primitive_action = 'push' if np.random.randint(0, 2) == 0 else 'grasp'

        # Get pixel location and rotation with highest affordance prediction (rotation, y, x)
        if self.primitive_action == 'push':
            self.compute_action(explore_actions, self.push_predictions)
        elif self.primitive_action == 'grasp':
            self.compute_action(explore_actions, self.grasp_predictions)
        elif self.primitive_action == 'place':
            self.compute_action(explore_actions, self.place_predictions)
        else:
            raise NotImplementedError('Primitive action type {} is not implemented'.format(self.primitive_action))

        # Save predicted confidence value
        self.trainer.predicted_value_log.append([self.predicted_value])
        self.logger.write_to_log('predicted-value', self.trainer.predicted_value_log)

    def compute_action(self, explore_actions, predictions):
        if explore_actions:
            maximas = utils.k_largest_index_argpartition(predictions, k=10)
            self.best_pix_ind = maximas[np.random.choice(maximas.shape[0])]
        else:
            self.best_pix_ind = np.unravel_index(np.argmax(predictions), predictions.shape)

        self.predicted_value = predictions[self.best_pix_ind[0], self.best_pix_ind[1], self.best_pix_ind[2]]

    def agent(self):
        """
        Parallel thread to process network output and execute actions
        """
        while not self.shutdown_called and self.trainer.iteration <= self.max_iter:
            if self.execute_action:
                # Select action based on policy
                self.policy()

                # Compute 3D position of pixel
                logging.info(
                    'Action: %s at (%d, %d, %d)' % (
                        self.primitive_action, self.best_pix_ind[0], self.best_pix_ind[1], self.best_pix_ind[2]))
                best_rotation_angle = np.deg2rad(self.best_pix_ind[0] * (360.0 / self.trainer.model.num_rotations))
                best_pix_x = self.best_pix_ind[2]
                best_pix_y = self.best_pix_ind[1]
                primitive_position = [best_pix_x * self.heightmap_resolution + self.workspace_limits[0][0],
                                      best_pix_y * self.heightmap_resolution + self.workspace_limits[1][0],
                                      self.depth_heightmap[best_pix_y][best_pix_x] + self.workspace_limits[2][0]]

                # If pushing, adjust start position, and make sure z value is safe and not too low
                if self.primitive_action == 'push' or self.primitive_action == 'place':
                    finger_width = 0.02
                    safe_kernel_width = int(np.round((finger_width / 2) / self.heightmap_resolution))
                    local_region = self.depth_heightmap[
                                   max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1,
                                                                              self.depth_heightmap.shape[0]),
                                   max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1,
                                                                              self.depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = self.workspace_limits[2][0]
                    else:
                        safe_z_position = np.max(local_region) + self.workspace_limits[2][0]
                    primitive_position[2] = safe_z_position

                # Save executed primitive
                if self.primitive_action == 'push':
                    self.trainer.executed_action_log.append(
                        [0, self.best_pix_ind[0], self.best_pix_ind[1], self.best_pix_ind[2]])  # 0 - push
                elif self.primitive_action == 'grasp':
                    self.trainer.executed_action_log.append(
                        [1, self.best_pix_ind[0], self.best_pix_ind[1], self.best_pix_ind[2]])  # 1 - grasp
                elif self.primitive_action == 'place':
                    self.trainer.executed_action_log.append(
                        [2, self.best_pix_ind[0], self.best_pix_ind[1], self.best_pix_ind[2]])  # 2 - place
                self.logger.write_to_log('executed-action', self.trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                grasp_pred_vis = viz.get_prediction_vis(self.grasp_predictions, self.color_heightmap,
                                                        self.best_pix_ind, 'grasp')
                imgs = torch.from_numpy(grasp_pred_vis).permute(2, 0, 1)
                self.tb.add_image('grasp_pred', imgs, self.trainer.iteration)

                # grasp_pred_vis = viz.get_prediction_full_vis(self.grasp_predictions, self.color_heightmap, self.best_pix_ind)
                # imgs = torch.from_numpy(grasp_pred_vis).permute(2, 0, 1)
                # self.tb.add_image('grasp_pred_full', imgs, self.trainer.iteration)

                if self.push_enabled:
                    push_pred_vis = viz.get_prediction_vis(self.push_predictions, self.color_heightmap,
                                                           self.best_pix_ind, 'push')
                    imgs = torch.from_numpy(push_pred_vis).permute(2, 0, 1)
                    self.tb.add_image('push_pred', imgs, self.trainer.iteration)

                if self.place_enabled:
                    place_pred_vis = viz.get_prediction_vis(self.place_predictions, self.color_heightmap,
                                                            self.best_pix_ind, 'place')
                    imgs = torch.from_numpy(place_pred_vis).permute(2, 0, 1)
                    self.tb.add_image('place_pred', imgs, self.trainer.iteration)

                if self.save_visualizations:
                    if self.primitive_action == 'push':
                        self.logger.save_visualizations(self.trainer.iteration, push_pred_vis, 'push')
                    elif self.primitive_action == 'grasp':
                        self.logger.save_visualizations(self.trainer.iteration, grasp_pred_vis, 'grasp')
                    elif self.primitive_action == 'place':
                        self.logger.save_visualizations(self.trainer.iteration, place_pred_vis, 'place')

                # Initialize variables that influence reward
                push_success = False
                grasp_success = False
                place_success = False

                # Execute primitive
                pool = concurrent.futures.ThreadPoolExecutor()
                try:
                    if self.primitive_action == 'push':
                        future = pool.submit(self.robot.push, primitive_position, best_rotation_angle)
                        push_success = future.result(timeout=60)
                        logging.info('Push successful: %r' % push_success)
                    elif self.primitive_action == 'grasp':
                        future = pool.submit(self.robot.grasp, primitive_position, best_rotation_angle)
                        grasp_success = future.result(timeout=60)
                        logging.info('Grasp successful: %r' % grasp_success)
                    elif self.primitive_action == 'place':
                        future = pool.submit(self.robot.place, primitive_position, best_rotation_angle)
                        place_success = future.result(timeout=60)
                        logging.info('Place successful: %r' % place_success)
                except concurrent.futures.TimeoutError:
                    logging.error('Robot execution timeout!')
                    self.mission_complete = False
                else:
                    self.mission_complete = True

                # Save information for next training step
                self.prev_color_heightmap = self.color_heightmap.copy()
                self.prev_depth_heightmap = self.depth_heightmap.copy()
                self.prev_grasp_success = grasp_success
                self.prev_push_success = push_success
                self.prev_place_success = place_success
                self.prev_primitive_action = self.primitive_action
                self.prev_best_pix_ind = self.best_pix_ind

                self.execute_action = False
            else:
                time.sleep(0.1)

    def compute_reward(self, change_detected, stack_height):
        # Compute current reward
        current_reward = 0
        if self.prev_primitive_action == 'push' and self.prev_push_success:
            if change_detected:
                if self.reward_type == 3:
                    current_reward = 0.75
                else:
                    current_reward = 0.5
            else:
                self.prev_push_success = False
        elif self.prev_primitive_action == 'grasp' and self.prev_grasp_success:
            if self.reward_type < 4:
                if (self.place_enabled and stack_height >= self.prev_stack_height) or (not self.place_enabled):
                    current_reward = 1.0
                else:
                    self.prev_grasp_success = False
            elif self.reward_type == 4:
                if self.place_enabled:
                    if stack_height >= self.prev_stack_height:
                        current_reward = 1.0
                    else:
                        self.prev_grasp_success = False
                        current_reward = -0.5
                else:
                    current_reward = 1.0
        elif self.prev_primitive_action == 'place' and self.prev_place_success:
            if stack_height > self.prev_stack_height:
                current_reward = self.place_reward_scale * stack_height
            else:
                self.prev_place_success = False

        # Compute future reward
        if self.place_enabled and not change_detected and not self.prev_grasp_success and not self.prev_place_success:
            future_reward = 0
        elif not self.place_enabled and not change_detected and not self.prev_grasp_success:
            future_reward = 0
        elif self.reward_type > 1 and current_reward == 0:
            future_reward = 0
        else:
            future_reward = self.predicted_value
        expected_reward = current_reward + self.future_reward_discount * future_reward

        return expected_reward, current_reward, future_reward

    def reward_function(self):
        # Detect changes
        depth_diff = abs(self.depth_heightmap - self.prev_depth_heightmap)
        depth_diff[np.isnan(depth_diff)] = 0
        depth_diff[depth_diff > 0.3] = 0
        depth_diff[depth_diff < 0.01] = 0
        depth_diff[depth_diff > 0] = 1
        change_threshold = 300
        change_value = np.sum(depth_diff)
        change_detected = change_value > change_threshold or self.prev_grasp_success
        logging.info('Change detected: %r (value: %d)' % (change_detected, change_value))

        if change_detected:
            if self.prev_primitive_action == 'push':
                self.no_change_count[0] = 0
            elif self.prev_primitive_action == 'grasp' or self.prev_primitive_action == 'place':
                self.no_change_count[1] = 0
        else:
            if self.prev_primitive_action == 'push':
                self.no_change_count[0] += 1
            elif self.prev_primitive_action == 'grasp':
                self.no_change_count[1] += 1

        # Check stack height
        img_median = ndimage.median_filter(self.depth_heightmap, size=5)
        max_z = np.max(img_median)
        if max_z <= 0.069:
            stack_height = 1
        elif (max_z > 0.069) and (max_z <= 0.11):
            stack_height = 2
        elif (max_z > 0.11) and (max_z <= 0.156):
            stack_height = 3
        elif (max_z > 0.156) and (max_z <= 0.21):
            stack_height = 4
        else:
            stack_height = 0

        if self.place_enabled:
            logging.info('Current stack height is {}'.format(stack_height))
            self.tb.add_scalar('stack_height', stack_height, self.trainer.iteration)

        # Compute reward
        expected_reward, current_reward, future_reward = self.compute_reward(change_detected, stack_height)

        logging.info('Current reward: %f' % current_reward)
        logging.info('Future reward: %f' % future_reward)
        logging.info('Expected reward: %f + %f x %f = %f' % (
            current_reward, self.future_reward_discount, future_reward, expected_reward))

        self.prev_stack_height = stack_height

        return expected_reward, current_reward

    def experience_replay(self, prev_reward_value):
        """
         Sample a reward value from the same action as the current one which differs from the most recent reward value
         to reduce the chance of catastrophic forgetting
        """
        sample_primitive_action = self.prev_primitive_action
        if sample_primitive_action == 'push':
            sample_primitive_action_id = 0
            sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5
        elif sample_primitive_action == 'grasp':
            sample_primitive_action_id = 1
            sample_reward_value = 0 if prev_reward_value == 1 else 1
        elif sample_primitive_action == 'place':
            sample_primitive_action_id = 2
            sample_reward_value = 0 if prev_reward_value >= 1 else 1
        else:
            raise NotImplementedError(
                'ERROR: {} action is not yet supported in experience replay'.format(sample_primitive_action))

        # Get samples of the same primitive but with different results
        sample_ind = np.argwhere(np.logical_and(
            np.asarray(self.trainer.reward_value_log)[1:self.trainer.iteration, 0] == sample_reward_value,
            np.asarray(self.trainer.executed_action_log)[1:self.trainer.iteration, 0] == sample_primitive_action_id))

        if sample_ind.size > 0:

            # Find sample with highest surprise value
            sample_surprise_values = np.abs(np.asarray(self.trainer.predicted_value_log)[sample_ind[:, 0]] -
                                            np.asarray(self.trainer.label_value_log)[sample_ind[:, 0]])
            sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
            sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
            pow_law_exp = 2
            rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
            sample_iteration = sorted_sample_ind[rand_sample_ind]
            logging.info('Experience replay: iteration %d (surprise value: %f)' % (
                sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

            # Load sample RGB-D heightmap
            sample_color_heightmap = cv2.imread(
                os.path.join(self.logger.color_heightmaps_directory, '%06d.0.color.png' % sample_iteration))
            sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
            sample_depth_heightmap = cv2.imread(
                os.path.join(self.logger.depth_heightmaps_directory, '%06d.0.depth.png' % sample_iteration), -1)
            sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000

            # Compute forward pass with sample
            with torch.no_grad():
                sample_push_predictions, sample_grasp_predictions, sample_place_predictions = self.trainer.forward(
                    sample_color_heightmap, sample_depth_heightmap, is_volatile=True)

            # Get labels for sample and backpropagate
            sample_best_pix_ind = (np.asarray(self.trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
            self.trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action,
                                  sample_best_pix_ind, self.trainer.label_value_log[sample_iteration], self.filter_type)

            # Recompute prediction value and label for replay buffer
            if sample_primitive_action == 'push':
                self.trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
            elif sample_primitive_action == 'grasp':
                self.trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
            elif sample_primitive_action == 'place':
                self.trainer.predicted_value_log[sample_iteration] = [np.max(sample_place_predictions)]

        else:
            logging.info('Not enough prior training samples. Skipping experience replay.')

    def loop(self):
        """
        Main training/testing loop
        """
        # Init current mission
        self.mission_complete = False
        reset_trial = False

        # Make sure simulation is still stable (if not, reset simulation)
        if self.is_sim:
            self.robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = self.robot.get_camera_data()
        depth_img = depth_img * self.robot.cam_depth_scale  # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        self.color_heightmap, self.depth_heightmap = utils.get_heightmap(color_img, depth_img,
                                                                         self.robot.cam_intrinsics,
                                                                         self.robot.cam_pose, self.workspace_limits,
                                                                         self.heightmap_resolution)
        # Remove NaNs from the depth heightmap
        self.depth_heightmap[np.isnan(self.depth_heightmap)] = 0

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(self.depth_heightmap.shape)
        stuff_count[self.depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if self.is_sim and self.is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold:
            logging.info('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
            reset_trial = True

        # Reset simulation or pause real-world training if no change is detected for last 10 iterations
        if self.is_sim and self.no_change_count[0] + self.no_change_count[1] > 15:
            logging.info('No change is detected for last 15 iterations. Resetting simulation.')
            reset_trial = True

        if self.prev_stack_height >= self.goal_stack_height and self.place_enabled:
            logging.info('Stack completed. Repositioning objects.')
            reset_trial = True

        if not reset_trial:
            # Run forward pass with network to get affordances
            self.push_predictions, self.grasp_predictions, self.place_predictions = self.trainer.forward(
                self.color_heightmap, self.depth_heightmap, is_volatile=True)

            # Execute best primitive action on robot in another thread
            self.execute_action = True

            # Save RGB-D images and RGB-D heightmaps
            self.logger.save_images(self.trainer.iteration, color_img, depth_img, '0')
            self.logger.save_heightmaps(self.trainer.iteration, self.color_heightmap, self.depth_heightmap, '0')

        # Run training iteration in current thread (aka training thread)
        if self.prev_primitive_action is not None:

            # Compute training labels
            label_value, prev_reward_value = self.reward_function()

            # Backpropagate
            self.trainer.backprop(self.prev_color_heightmap, self.prev_depth_heightmap,
                                  self.prev_primitive_action, self.prev_best_pix_ind, label_value, self.filter_type)

            # Save training labels and reward
            self.trainer.label_value_log.append([label_value])
            self.trainer.reward_value_log.append([prev_reward_value])
            self.trainer.grasp_success_log.append([int(self.prev_grasp_success)])
            self.logger.write_to_log('label-value', self.trainer.label_value_log)
            self.logger.write_to_log('reward-value', self.trainer.reward_value_log)
            self.logger.write_to_log('grasp-success', self.trainer.grasp_success_log)
            if self.push_enabled:
                self.trainer.push_success_log.append([int(self.prev_push_success)])
                self.logger.write_to_log('push-success', self.trainer.push_success_log)
            if self.place_enabled:
                self.trainer.place_success_log.append([int(self.prev_place_success)])
                self.logger.write_to_log('place-success', self.trainer.place_success_log)

            # Save to tensorboard
            self.tb.add_scalar('loss', self.trainer.running_loss.mean(), self.trainer.iteration)
            if self.prev_primitive_action == 'grasp':
                self.tb.add_scalar('success-rate/grasp', self.prev_grasp_success, self.trainer.iteration)
            elif self.prev_primitive_action == 'push':
                self.tb.add_scalar('success-rate/push', self.prev_push_success, self.trainer.iteration)
            elif self.prev_primitive_action == 'place':
                self.tb.add_scalar('success-rate/place', self.prev_place_success, self.trainer.iteration)

            if not self.is_testing:
                # Adjust exploration probability
                if self.explore_type == 1:
                    self.explore_prob = max(0.5 * np.power(0.99994, self.trainer.iteration),
                                            0.1) if self.explore_rate_decay else 0.5
                elif self.explore_type == 2:
                    f = (1.0 - np.exp((-self.trainer.running_loss.mean()) / self.LAE_sigma)) \
                        / (1.0 + np.exp((-self.trainer.running_loss.mean()) / self.LAE_sigma))
                    self.explore_prob = self.LAE_beta * f + (1 - self.LAE_beta) * self.explore_prob

                # Check for progress counting inconsistencies
                if len(self.trainer.reward_value_log) < self.trainer.iteration - 2:
                    logging.warning(
                        'WARNING POSSIBLE CRITICAL ERROR DETECTED: log data index and trainer.iteration out of sync!!! '
                        'Experience Replay may break! '
                        'Check code for errors in indexes, continue statements etc.')

                if not self.experience_replay_disabled:
                    # Do sampling for experience replay
                    self.experience_replay(prev_reward_value)

                # Save model snapshot
                self.logger.save_backup_model(self.trainer.model)
                if self.trainer.iteration % 1000 == 0:
                    self.logger.save_model(self.trainer.iteration, self.trainer.model)
                    self.trainer.model.to(self.trainer.device)

        if not reset_trial:
            # Sync both action thread and training thread
            while self.execute_action:
                time.sleep(0.1)

            if self.mission_complete:
                logging.info('Mission complete')
            else:
                logging.warning('Robot execution failed. Restarting simulation..')
                self.robot.restart_sim()

        if reset_trial:
            if self.is_sim:
                self.robot.restart_sim()
                self.robot.add_objects()
            else:
                time.sleep(30)

            if self.is_testing:  # If at end of test run, re-load original weights (before test run)
                self.trainer.model.load_state_dict(torch.load(self.snapshot_file))

            self.trainer.task_complete_log.append([self.trainer.iteration])
            self.logger.write_to_log('task_complete', self.trainer.task_complete_log)
            self.tb.add_scalar('task_complete', self.trainer.iteration - self.last_task_complete, len(self.trainer.task_complete_log))

            self.last_task_complete = self.trainer.iteration
            self.no_change_count = [2, 2] if not self.is_testing else [0, 0]
            self.prev_stack_height = 0
            self.prev_primitive_action = None

    def run(self):
        agent_thread = threading.Thread(target=self.agent)
        agent_thread.daemon = True
        agent_thread.start()
        while self.trainer.iteration <= self.max_iter:
            logging.info('\n%s iteration: %d' % ('Testing' if self.is_testing else 'Training', self.trainer.iteration))

            # Main loop
            iteration_time_0 = time.time()
            self.loop()
            if self.mission_complete:
                self.trainer.iteration += 1
            iteration_time_1 = time.time()
            logging.info('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))

            # Check for number of test trails completed
            if self.is_testing and len(self.trainer.task_complete_log) >= self.max_test_trials:
                break
        self.shutdown_called = True
        agent_thread.join()

    def teardown(self):
        self.robot.shutdown()
        del self.trainer, self.robot
        torch.cuda.empty_cache()


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn manipulation actions with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=True,
                        help='run in simulation?')
    parser.add_argument('--sim_port', dest='sim_port', type=int, action='store', default=19997,
                        help='port for simulation')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='simulation/objects/mixed_shapes',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,
                        help='number of objects to add to simulation')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',
                        default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=123,
                        help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--network', dest='network', action='store', default='grconvnet4',
                        help='Neural network architecture choice, options are grconvnet, efficientnet, denseunet')
    parser.add_argument('--num_rotations', dest='num_rotations', type=int, action='store', default=16)
    parser.add_argument('--push_enabled', dest='push_enabled', action='store_true', default=False)
    parser.add_argument('--place_enabled', dest='place_enabled', action='store_true', default=False)
    parser.add_argument('--reward_type', dest='reward_type', type=int, action='store', default=2)
    parser.add_argument('--filter_type', dest='filter_type', type=int, action='store', default=4)
    parser.add_argument('--experience_replay_disabled', dest='experience_replay_disabled', action='store_true',
                        default=False, help='disable prioritized experience replay')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store',
                        default=0.5)
    parser.add_argument('--place_reward_scale', dest='place_reward_scale', type=float, action='store', default=1.0)
    parser.add_argument('--goal_stack_height', dest='goal_stack_height', type=int, action='store', default=4)
    parser.add_argument('--explore_actions', dest='explore_actions', type=int, action='store', default=1)
    parser.add_argument('--explore_type', dest='explore_type', type=int, action='store', default=1)
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=True)
    parser.add_argument('--max_iter', dest='max_iter', action='store', type=int, default=50000,
                        help='max iter for training')

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,
                        help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='')
    parser.add_argument('--test_preset_dir', dest='test_preset_dir', action='store', default='simulation/test-cases/')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,
                        help='save visualizations of model predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()

    if args.is_testing and args.test_preset_cases:
        preset_files = os.listdir(args.test_preset_dir)
        preset_files = [os.path.abspath(os.path.join(args.test_preset_dir, filename)) for filename in preset_files]
        preset_files = sorted(preset_files)
        args.continue_logging = True
        for idx, preset_file in enumerate(preset_files):
            logging.info('Running test {}'.format(preset_file))
            args.test_preset_file = preset_file
            args.num_obj = 10
            args.logging_directory = args.snapshot_file.split('/')[0] + '/' + args.snapshot_file.split('/')[
                1] + '/preset-test/' + re.findall("\d+", args.snapshot_file.split('/')[3])[0] + '/' + str(idx)

            task = LearnManipulation(args)
            task.run()
            task.teardown()
    else:
        task = LearnManipulation(args)
        task.run()
        task.teardown()

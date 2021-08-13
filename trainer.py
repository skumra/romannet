import logging
import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from models.network import ManipulationNet


class Trainer:
    def __init__(self, network, force_cpu, push_enabled, place_enabled, num_rotations):

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            logging.info("CUDA detected. Running with GPU acceleration.")
            use_cuda = True
        elif force_cpu:
            logging.info("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            use_cuda = False
        else:
            logging.info("CUDA is *NOT* detected. Running with only CPU.")
            use_cuda = False

        if use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ManipulationNet for deep reinforcement learning
        self.push_enabled = push_enabled
        self.place_enabled = place_enabled
        self.model = ManipulationNet(network, self.device, self.push_enabled, self.place_enabled, num_rotations)
        self.model = self.model.to(self.device)

        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False)
        self.criterion = self.criterion.to(self.device)

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0
        self.loss_value = 0
        self.running_loss = np.ones(10)

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.grasp_success_log = []
        self.predicted_value_log = []
        self.is_exploit_log = []
        self.task_complete_log = []
        if self.push_enabled:
            self.push_success_log = []
        if self.place_enabled:
            self.place_success_log = []

    def load_snapshot(self, snapshot_file):
        """
        Load pre-trained model
        """
        self.model.load_state_dict(torch.load(snapshot_file))
        logging.info('Pre-trained model snapshot loaded from: %s' % snapshot_file)

    def preload(self, transitions_directory):
        """
        Pre-load execution info and RL variables
        """
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'),
                                              delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration, 1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'),
                                              delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration, 1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.grasp_success_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-success.log.txt'), delimiter=' ')
        self.grasp_success_log = self.grasp_success_log[0:self.iteration]
        self.grasp_success_log.shape = (self.iteration, 1)
        self.grasp_success_log = self.grasp_success_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration, 1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.task_complete_log = np.loadtxt(os.path.join(transitions_directory, 'task_complete.log.txt'), delimiter=' ')
        self.task_complete_log.shape = (self.task_complete_log.shape[0], 1)
        self.task_complete_log = self.task_complete_log.tolist()
        if self.push_enabled:
            self.push_success_log = np.loadtxt(os.path.join(transitions_directory, 'push-success.log.txt'),
                                               delimiter=' ')
            self.push_success_log = self.push_success_log[0:self.iteration]
            self.push_success_log.shape = (self.iteration, 1)
            self.push_success_log = self.push_success_log.tolist()
        if self.place_enabled:
            self.place_success_log = np.loadtxt(os.path.join(transitions_directory, 'place-success.log.txt'),
                                                delimiter=' ')
            self.place_success_log = self.place_success_log[0:self.iteration]
            self.place_success_log.shape = (self.iteration, 1)
            self.place_success_log = self.place_success_log.tolist()

    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=None):
        """
        Compute forward pass through model to compute affordances/Q
        """
        # Pre-process
        data = self.model.pre_process(color_heightmap, depth_heightmap)

        # Pass input data through model
        output_prob = self.model.forward(data, is_volatile, specific_rotation)

        # Post process
        push_predictions, grasp_predictions, place_predictions = self.model.post_process(data, output_prob)

        return push_predictions, grasp_predictions, place_predictions

    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, filter_type):
        """
        Compute labels and backpropagate
        """
        # Compute labels
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1

        if filter_type == 1:
            blur_kernel = np.ones((5, 5), np.float32)/25
            action_area = cv2.filter2D(action_area, -1, blur_kernel)
        elif filter_type == 2:
            action_area = cv2.GaussianBlur(action_area, (3, 3), 0)
        elif filter_type == 3:
            action_area = cv2.GaussianBlur(action_area, (5, 5), 0)

        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

        _label = Variable(torch.from_numpy(label).float().to(self.device))
        _label_weights = Variable(torch.from_numpy(label_weights).float().to(self.device), requires_grad=False)

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        if primitive_action == 'push':

            # Do forward pass with specified rotation (to save gradients)
            self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
            loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320), _label) * _label_weights
            loss = loss.sum()
            loss.backward()
            self.loss_value = loss.cpu().data.numpy()

        elif primitive_action == 'grasp':

            # Do forward pass with specified rotation (to save gradients)
            self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
            loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), _label) * _label_weights
            loss = loss.sum()
            loss.backward()
            self.loss_value = loss.cpu().data.numpy()

            opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations

            self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)

            loss = self.criterion(self.model.output_prob[0][1].view(1, 320, 320), _label) * _label_weights

            loss = loss.sum()
            loss.backward()
            self.loss_value += loss.cpu().data.numpy()

            self.loss_value = self.loss_value / 2

        elif primitive_action == 'place':

            # Do forward pass with specified rotation (to save gradients)
            self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
            loss = self.criterion(self.model.output_prob[0][2].view(1, 320, 320), _label) * _label_weights
            loss = loss.sum()
            loss.backward()
            self.loss_value = loss.cpu().data.numpy()

            opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations

            self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)
            loss = self.criterion(self.model.output_prob[0][2].view(1, 320, 320), _label) * _label_weights
            loss = loss.sum()
            loss.backward()
            self.loss_value += loss.cpu().data.numpy()

            self.loss_value = self.loss_value / 2

        logging.info('Training loss: %f' % self.loss_value)
        self.optimizer.step()
        self.running_loss[:-1] = self.running_loss[1:]
        self.running_loss[-1] = self.loss_value

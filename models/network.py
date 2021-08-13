#!/usr/bin/env python

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.autograd import Variable

from models.denseunet import DenseUNet
from models.grconvnet import GenerativeResnet
from models.grconvnet3 import GenerativeResnet3
from models.grconvnet4 import GenerativeResnet4


class ManipulationNet(nn.Module):

    def __init__(self, network, device, push_enabled, place_enabled, num_rotations):
        super(ManipulationNet, self).__init__()
        self.device = device
        self.push_enabled = push_enabled
        self.place_enabled = place_enabled
        self.net = None
        self.preprocess_input = None

        # Initialize network
        if network == 'grconvnet':
            self.net = GenerativeResnet()
        elif network == 'grconvnet3':
            # if self.push_enabled:
            self.push_net = GenerativeResnet3()
            self.grasp_net = GenerativeResnet3()
            self.place_net = GenerativeResnet3()
        elif network == 'grconvnet4':
            # if self.push_enabled:
            self.push_net = GenerativeResnet4()
            self.grasp_net = GenerativeResnet4()
            self.place_net = GenerativeResnet4()
        elif network == 'denseunet':
            self.net = DenseUNet()
        elif network == 'efficientunet':
            encoder = 'efficientnet-b4'
            encoder_weights = 'imagenet'
            self.preprocess_input = get_preprocessing_fn(encoder, pretrained=encoder_weights)
            self.push_net = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=4)
            self.grasp_net = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=4)
            self.place_net = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=4)
        else:
            raise NotImplementedError('Network type {} is not implemented'.format(network))

        self.num_rotations = num_rotations

        # Initialize variables
        self.padding_width = 0
        self.output_prob = []

    def pre_process(self, color_heightmap, depth_heightmap):

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        self.padding_width = int((diag_length - color_heightmap.shape[0]) / 2)
        color_heightmap_r = np.pad(color_heightmap[:, :, 0], self.padding_width, 'constant', constant_values=0)
        color_heightmap_r.shape = (color_heightmap_r.shape[0], color_heightmap_r.shape[1], 1)
        color_heightmap_g = np.pad(color_heightmap[:, :, 1], self.padding_width, 'constant', constant_values=0)
        color_heightmap_g.shape = (color_heightmap_g.shape[0], color_heightmap_g.shape[1], 1)
        color_heightmap_b = np.pad(color_heightmap[:, :, 2], self.padding_width, 'constant', constant_values=0)
        color_heightmap_b.shape = (color_heightmap_b.shape[0], color_heightmap_b.shape[1], 1)
        input_color_image = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
        input_depth_image = np.pad(depth_heightmap, self.padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = input_color_image.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        if self.preprocess_input is not None:
            input_color_image = self.preprocess_input(input_color_image)

        # Pre-process depth image (normalize)
        image_mean = 0.01
        image_std = 0.03
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], 1)
        input_depth_image = (input_depth_image - image_mean) / image_std

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
        input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (
        input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

        data = torch.cat((input_color_data, input_depth_data), dim=1)

        return data

    def post_process(self, data, output_prob):

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,
                                   int(self.padding_width):int(data.shape[2] - self.padding_width),
                                   int(self.padding_width):int(data.shape[2] - self.padding_width)]
                grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:, 0,
                                    int(self.padding_width):int(data.shape[2] - self.padding_width),
                                    int(self.padding_width):int(data.shape[2] - self.padding_width)]
                place_predictions = output_prob[rotate_idx][2].cpu().data.numpy()[:, 0,
                                    int(self.padding_width):int(data.shape[2] - self.padding_width),
                                    int(self.padding_width):int(data.shape[2] - self.padding_width)]
            else:
                push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,
                                                                     int(self.padding_width):int(
                                                                         data.shape[2] - self.padding_width),
                                                                     int(self.padding_width):int(
                                                                         data.shape[2] - self.padding_width)]), axis=0)
                grasp_predictions = np.concatenate((grasp_predictions,
                                                    output_prob[rotate_idx][1].cpu().data.numpy()[:, 0,
                                                    int(self.padding_width):int(data.shape[2] - self.padding_width),
                                                    int(self.padding_width):int(data.shape[2] - self.padding_width)]),
                                                   axis=0)
                place_predictions = np.concatenate((place_predictions,
                                                    output_prob[rotate_idx][2].cpu().data.numpy()[:, 0,
                                                    int(self.padding_width):int(data.shape[2] - self.padding_width),
                                                    int(self.padding_width):int(data.shape[2] - self.padding_width)]),
                                                   axis=0)

        return push_predictions, grasp_predictions, place_predictions

    def forward(self, data, is_volatile=False, specific_rotation=-1):
        if is_volatile:
            torch.set_grad_enabled(False)
            output_prob = []

            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).to(self.device), data.size())

                # Rotate images clockwise
                rotate_data = F.grid_sample(Variable(data, volatile=True).to(self.device), flow_grid_before, mode='nearest')

                # Compute features
                if self.net is not None:
                    push_feat, grasp_feat, place_feat = self.net.forward(rotate_data)
                else:
                    push_feat = self.push_net.forward(rotate_data)
                    grasp_feat = self.grasp_net.forward(rotate_data)
                    place_feat = self.place_net.forward(rotate_data)

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                               [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).to(self.device), push_feat.data.size())

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append([F.grid_sample(push_feat, flow_grid_after, mode='nearest'),
                                    F.grid_sample(grasp_feat, flow_grid_after, mode='nearest'),
                                    F.grid_sample(place_feat, flow_grid_after, mode='nearest')])
            torch.set_grad_enabled(True)

            return output_prob

        else:
            self.output_prob = []

            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).to(self.device), data.size())

            # Rotate images clockwise
            rotate_data = F.grid_sample(Variable(data, requires_grad=False).to(self.device), flow_grid_before, mode='nearest')

            # Compute features
            if self.net is not None:
                push_feat, grasp_feat, place_feat = self.net.forward(rotate_data)
            else:
                push_feat = self.push_net.forward(rotate_data)
                grasp_feat = self.grasp_net.forward(rotate_data)
                place_feat = self.place_net.forward(rotate_data)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).to(self.device), push_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([F.grid_sample(push_feat, flow_grid_after, mode='nearest'),
                                     F.grid_sample(grasp_feat, flow_grid_after, mode='nearest'),
                                     F.grid_sample(place_feat, flow_grid_after, mode='nearest')])

            return self.output_prob

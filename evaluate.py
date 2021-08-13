#!/usr/bin/env python

import argparse
import logging
import os
import re

import numpy as np


def evaluate(session_directory, num_obj_complete):
    # Parse data from session (action executed, reward values)
    transitions_directory = os.path.join(session_directory, 'transitions')
    executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
    max_iteration = executed_action_log.shape[0]
    executed_action_log = executed_action_log[0:max_iteration, :]
    reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
    reward_value_log = reward_value_log[0:max_iteration]
    clearance_log = np.loadtxt(os.path.join(transitions_directory, 'task_complete.log.txt'), delimiter=' ')
    clearance_log = np.unique(clearance_log)
    max_trials = len(clearance_log)
    clearance_log = np.concatenate((np.asarray([0]), clearance_log), axis=0).astype(int)

    # Count number of pushing/grasping actions before completion
    num_actions_before_completion = clearance_log[1:(max_trials + 1)] - clearance_log[0:(max_trials)]

    grasp_success_rate = np.zeros(max_trials)
    grasp_num_success = np.zeros(max_trials)
    grasp_to_push_ratio = np.zeros(max_trials)
    for trial_idx in range(1, len(clearance_log)):
        # Get actions and reward values for current trial
        tmp_executed_action_log = executed_action_log[clearance_log[trial_idx - 1]:clearance_log[trial_idx], 0]
        tmp_reward_value_log = reward_value_log[clearance_log[trial_idx - 1]:clearance_log[trial_idx]]

        # Get indices of pushing and grasping actions for current trial
        tmp_grasp_attempt_ind = np.argwhere(tmp_executed_action_log == 1)
        tmp_push_attempt_ind = np.argwhere(tmp_executed_action_log == 0)

        grasp_to_push_ratio[trial_idx - 1] = float(len(tmp_grasp_attempt_ind)) / float(len(tmp_executed_action_log))

        # Count number of times grasp attempts were successful
        # Reward value for successful grasping is anything larger than 0.5
        tmp_num_grasp_success = np.sum(tmp_reward_value_log[
                                           tmp_grasp_attempt_ind] >= 0.5)

        grasp_num_success[trial_idx - 1] = tmp_num_grasp_success
        grasp_success_rate[trial_idx - 1] = float(tmp_num_grasp_success) / float(len(tmp_grasp_attempt_ind))

    # Which trials reached task completion?
    valid_clearance = grasp_num_success >= num_obj_complete

    # Display results
    clearance = float(np.sum(valid_clearance)) / float(max_trials) * 100
    logging.info('Average %% clearance: %2.1f' % clearance)

    grasp_success = np.mean(grasp_success_rate[valid_clearance]) * 100
    logging.info('Average %% grasp success per clearance: %2.1f' % grasp_success)

    action_efficiency = 100 * np.mean(
        np.divide(float(num_obj_complete), num_actions_before_completion[valid_clearance]))
    logging.info('Average %% action efficiency: %2.1f' % action_efficiency)

    grasp_to_push_ratio = np.mean(grasp_to_push_ratio[valid_clearance]) * 100
    logging.info('Average grasp to push ratio: %2.1f' % grasp_to_push_ratio)

    return clearance, grasp_success, action_efficiency, grasp_to_push_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the performance all test sessions.')
    parser.add_argument('--session_directory', dest='session_directory', action='store', type=str,
                        help='path to session directory for which to measure performance')
    parser.add_argument('--test_type', dest='test_type', action='store', type=str, default='preset',
                        help='type of test to evaluate. (random/preset)')
    parser.add_argument('--num_obj_complete', dest='num_obj_complete', action='store', type=int, default=10,
                        help='number of objects picked before considering task complete')
    args = parser.parse_args()

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(args.session_directory, 'evaluate'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    if args.test_type == 'random':
        session_directory = args.session_directory
        num_obj_complete = args.num_obj_complete
        evaluate(session_directory, num_obj_complete)
    elif args.test_type == 'preset':
        num_obj_presets = [4, 5, 3, 5, 5, 6, 3, 6, 6, 5, 4]
        preset_files = os.listdir(args.session_directory)
        preset_files = [os.path.abspath(os.path.join(args.session_directory, filename)) for filename in preset_files if
                        os.path.isdir(os.path.join(args.session_directory, filename))]
        preset_files = sorted(preset_files)

        avg_clearance = 0
        avg_grasp_success = 0
        avg_action_efficiency = 0
        avg_grasp_to_push_ratio = 0
        valid_clearance_count = 0
        complete_clearance_count = 0

        for idx, preset_file in enumerate(preset_files):
            logging.info('Preset {}: {}'.format(idx, preset_file))
            session_directory = preset_file
            m = re.search(r'\d+$', session_directory)
            num_obj_complete = num_obj_presets[int(m.group())]
            clearance, grasp_success, action_efficiency, grasp_to_push_ratio = evaluate(session_directory,
                                                                                        num_obj_complete)
            avg_clearance += clearance
            if clearance:
                avg_grasp_success += grasp_success
                avg_action_efficiency += action_efficiency
                avg_grasp_to_push_ratio += grasp_to_push_ratio
                valid_clearance_count += 1
            if clearance == 100:
                complete_clearance_count += 1

        logging.info('Summary')
        logging.info('Scenarios 100 %% complete: %d' % complete_clearance_count)
        logging.info('Overall average %% clearance: %2.1f' % (avg_clearance / (idx + 1)))
        logging.info('Overall average %% grasp success per clearance: %2.1f' % (avg_grasp_success / valid_clearance_count))
        logging.info('Overall average %% action efficiency: %2.1f' % (avg_action_efficiency / valid_clearance_count))
        logging.info('Overall average grasp to push ratio: %2.1f' % (avg_grasp_to_push_ratio / valid_clearance_count))
    else:
        raise NotImplementedError('Test type {} is not implemented'.format(args.test_type))

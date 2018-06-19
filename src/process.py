from __future__ import print_function
import os
import random
import math
import numpy as np
import lightgbm as lgb
import pandas as pd
import subprocess
import datetime
import time
#USER = 'Zhiying'
USER = 'Heng'
from sklearn.datasets import load_svmlight_file

root = r'C:\source\github\mlsa18-pass-prediction' if os.name == 'nt' else r'/mnt/c/source/github/mlsa18-pass-prediction'
output_separator = '\t'

class Player(object):
    def __init__(self, player_id, x, y):
        self.player_id = player_id
        self.x = x
        self.y = y

class Pass(object):
    def __init__(self, pass_id, line_num, time_start, time_end, sender_id, receiver_id):
        self.pass_id = pass_id
        self.line_num = line_num
        self.time_start = time_start
        self.time_end = time_end
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.sender = None
        self.receiver = None
        self.players = {}
        
    def add_player(self, player_id, x, y):
        self.players[player_id] = Player(player_id, x, y)
        
    def to_features(self):
        if not self.__validate():
            return None
        features = []
        features.append(self.sender_id)
        features.append(self.receiver_id)
        features.append(self.time_start)
        features.append(self.time_end - self.time_start) # duration
        features.append(self.__get_sender_receiver_distance())
        features.append(self.sender.x)
        features.append(self.sender.y)
        features.append(self.receiver.x)
        features.append(self.receiver.y)
        return output_separator.join([str(feature) for feature in features])
        
    def features_generator(self, get_features=False):
        sender = self.players[self.sender_id]
        sender_friends = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if candidate_id != self.sender_id and self.__in_same_team(self.sender_id, candidate_id)}
        sender_opponents = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if not self.__in_same_team(self.sender_id, candidate_id)}
        sender_friends_distances = sorted([self.__get_distance(friend_id, self.sender_id) for friend_id in sender_friends.keys()])
        sender_opponent_distances = sorted([self.__get_distance(opponent_id, self.sender_id) for opponent_id in sender_opponents.keys()])
        sender_closest_friend_dist = sender_friends_distances[0]
        sender_closest_3_friends_dist = np.mean(sender_friends_distances[:3])
        sender_closest_opponent_dist = sender_opponent_distances[0]
        sender_closest_3_oppononents_dist = np.mean(sender_opponent_distances[:3])
        is_sender_left_team = self.__is_player_left_team(self.sender_id)
        sender_field = self.__get_player_field(self.sender_id, is_sender_left_team)
        is_sender_in_back_field = 1 if sender_field == 0 else 0
        is_sender_in_middle_field = 1 if sender_field == 1 else 0
        is_sender_in_front_field = 1 if sender_field == 2 else 0
        sender_to_offense_gate_dist, sender_to_defense_gate_dist = self.__player_to_gate_distance(self.sender_id, is_sender_left_team)

        sender_friends_to_offense_gate_dists = [self.__player_to_gate_distance(friend_id, is_sender_left_team)[0]
                                                for friend_id in sender_friends.keys()]
        sender_to_offense_gate_dist_rank_relative_to_friends = \
            sum([dist < sender_to_offense_gate_dist for dist in sender_friends_to_offense_gate_dists]) + 1
        sender_opponents_to_offense_gate_dists = [self.__player_to_gate_distance(opponent_id, is_sender_left_team)[0]
                                                  for opponent_id in sender_opponents.keys()]
        sender_to_offense_gate_dist_rank_relative_to_opponents = \
            sum([dist < sender_to_offense_gate_dist for dist in sender_opponents_to_offense_gate_dists]) + 1
        sender_to_top_sideline_dist_rank_relative_to_friends = \
            sum([friend.y > sender.y for friend in sender_friends.values()]) + 1
        sender_to_top_sideline_dist_rank_relative_to_opponents = \
            sum([opponent.y > sender.y for opponent in sender_opponents.values()]) + 1

        sender_team = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if self.__in_same_team(self.sender_id, candidate_id)}
        sender_team_to_offense_goal_line_dists = [
            self.__player_to_goal_line_distance(team_member_id, is_sender_left_team)[0] for team_member_id in
            sender_team.keys()]
        sender_team_to_defense_goal_line_dists = [
            self.__player_to_goal_line_distance(team_member_id, is_sender_left_team)[1] for team_member_id in
            sender_team.keys()]
        sender_team_closest_dist_to_offense_goal_line = \
            sorted(sender_team_to_offense_goal_line_dists)[0]
        sender_team_closest_dist_to_defense_goal_line_exclude_goalie = \
            sorted(sender_team_to_defense_goal_line_dists)[1]
        sender_team_closest_dist_to_top_sideline = \
            sorted([3400 - player.y for player in sender_team.values()])[0]
        sender_team_cloeset_dist_to_bottom_sideline = \
            sorted([player.y - (-3400) for player in sender_team.values()])[0]
        sender_team_median_dist_to_offense_goal_line = \
            np.median(sender_team_to_offense_goal_line_dists)
        sender_team_median_dist_to_top_sideline = \
            np.median([3400 - player.y for player in sender_team.values()])
    
        for player_id in self.players.keys():
            if player_id == self.sender_id: continue
            #sender = self.players[self.sender_id]
            player = self.players[player_id]
            
            friends = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if candidate_id != player_id and self.__in_same_team(player_id, candidate_id)}
            opponents = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if not self.__in_same_team(player_id, candidate_id)}

            label = 1 if player_id == self.receiver_id else 0
            is_in_same_team = 1 if self.sender_id in friends.keys() else 0
            distance = self.__get_distance(self.sender_id, player_id)
            friend_distances = sorted([self.__get_distance(friend_id, player_id) for friend_id in friends.keys()])
            opponent_distances = sorted([self.__get_distance(opponent_id, player_id) for opponent_id in opponents.keys()])
            
            is_player_left_team = self.__is_player_left_team(player_id)
            player_field = self.__get_player_field(player_id, is_player_left_team)
            is_player_in_back_field = 1 if player_field == 0 else 0
            is_player_in_middle_field = 1 if player_field == 1 else 0
            is_player_in_front_field = 1 if player_field == 2 else 0
            is_sender_player_in_same_field = 1 if sender_field == player_field else 0
            normalized_player_to_sender_x_diff = self.__normalized_player_to_sender_x_diff(player_id, is_sender_left_team)
            
            player_to_offense_gate_dist, player_to_defense_gate_dist = self.__player_to_gate_distance(player_id, is_player_left_team)
            
            opponent_to_line_dists = self.__get_min_opponent_dist_to_sender_player_line(player_id)

            ## here offense gate is related to the sender (i.e., the target gate of sender)
            player_to_offense_gate_of_sender_dist, player_to_defense_gate_of_sender_dist = \
                self.__player_to_gate_distance(player_id, is_sender_left_team)
            player_friends_to_offense_gate_dists = [self.__player_to_gate_distance(friend_id, is_sender_left_team)[0]
                                                    for friend_id in friends.keys()]
            player_to_offense_gate_dist_rank_relative_to_friends = \
                sum([dist < player_to_offense_gate_of_sender_dist for dist in player_friends_to_offense_gate_dists]) + 1
            player_opponents_to_offense_gate_dists = [self.__player_to_gate_distance(opponent_id, is_sender_left_team)[0]
                                                      for opponent_id in opponents.keys()]
            player_to_offense_gate_dist_rank_relative_to_opponents = \
                sum([dist < player_to_offense_gate_of_sender_dist for dist in player_opponents_to_offense_gate_dists]) + 1
            player_to_top_sideline_dist_rank_relative_to_friends = \
                sum([friend.y > player.y for friend in friends.values()]) + 1
            player_to_top_sideline_dist_rank_relative_to_opponents = \
                sum([opponent.y > player.y for opponent in opponents.values()]) + 1

            player_friends_to_sender_dists = {self.__get_distance(self.sender_id, friend_id)
                                              for friend_id in friends.keys() if friend_id != self.sender_id}
            player_to_sender_dist_rank_among_friends = sum([dist < distance for dist in player_friends_to_sender_dists]) + 1
            player_opponents_to_sender_dists = {self.__get_distance(self.sender_id, opponent_id)
                                                for opponent_id in opponents.keys() if opponent_id != self.sender_id}
            player_to_sender_dist_rank_among_opponents = sum([dist < distance for dist in player_opponents_to_sender_dists]) + 1

            ## passing angle
            dangerous_opponents = [opponent_id for opponent_id in opponents.keys()
                                   if self.__get_distance(self.sender_id, opponent_id) < distance and
                                   self.__get_distance(player_id, opponent_id) < distance]
            if len(dangerous_opponents) == 0:
                min_pass_angle = 90
            else:
                pass_angles = [self.__calculate_sender_player_opponent_angle(self.sender_id, player_id, opponent_id)
                               for opponent_id in dangerous_opponents]
                min_pass_angle = np.min(pass_angles)
            num_dangerous_opponents_along_passing_line = len(dangerous_opponents)

            ## closest friends/opponents features
            closest_friend_id = sorted(friends.keys(),
                                       key=lambda friend_id: self.__get_distance(player_id, friend_id))[0]
            closest_opponent_id = sorted(opponents.keys(),
                                         key=lambda opponent_id: self.__get_distance(player_id, opponent_id))[0]
            player_closest_friend_to_sender_dist = self.__get_distance(self.sender_id, closest_friend_id)
            player_closest_opponent_to_sender_dist = self.__get_distance(self.sender_id, closest_opponent_id)

            features = []
            features.append(self.pass_id)
            features.append(self.line_num)
            features.append(label)
            features.append(self.sender_id)
            features.append(player_id)
            features.append(is_in_same_team)
            features.append(self.time_start)
            #features.append(self.time_end - self.time_start) # duration
            #features.append(distance / (self.time_end - self.time_start + 0.0001)) # ball speed
            #features.append(sender.x)
            #features.append(sender.y)
            #features.append(player.x)
            #features.append(player.y)
            features.append(distance)
            features.append(opponent_to_line_dists[0]) # min_opponent_dist_to_sender_player_line
            features.append(opponent_to_line_dists[1]) # second_opponent_dist_to_sender_player_line
            features.append(opponent_to_line_dists[2]) # third_opponent_dist_to_sender_player_line
            features.append(is_sender_in_back_field)
            features.append(is_sender_in_middle_field)
            features.append(is_sender_in_front_field)
            features.append(is_player_in_back_field)
            features.append(is_player_in_middle_field)
            features.append(is_player_in_front_field)
            features.append(is_sender_player_in_same_field)
            features.append(sender_to_offense_gate_dist)
            features.append(sender_to_defense_gate_dist)
            features.append(player_to_offense_gate_dist)
            features.append(player_to_defense_gate_dist)
            features.append(normalized_player_to_sender_x_diff)
            features.append(1 if normalized_player_to_sender_x_diff >= 0 else 0) # is_player_in_offense_direction_relative_to_sender
            features.append(math.fabs(sender.y - player.y)) # abs_y_diff
            features.append(self.__is_start_of_game())
            features.append(self.__distance_to_center(self.sender_id)) # sender_to_center_distance
            features.append(self.__distance_to_center(player_id)) # player_to_center_distance
            features.append(1 if self.__distance_to_center(player_id) <= 915 else 0) # is_player_in_center_circle
            features.append(self.__is_goal_keeper(self.sender_id, is_sender_left_team)) # is_sender_goal_keeper
            features.append(self.__is_goal_keeper(player_id, is_player_left_team)) # is_player_goal_keeper
            features.append(sender_closest_friend_dist)
            features.append(sender_closest_3_friends_dist)
            features.append(sender_closest_opponent_dist)
            features.append(sender_closest_3_oppononents_dist)
            features.append(friend_distances[0]) # closest friend distance
            features.append(np.mean(friend_distances[:3])) # closest 3 friends avg distance
            features.append(opponent_distances[0]) # closest opponent distance
            features.append(np.mean(opponent_distances[:3])) # closest 3 opponent avg distance
            features.append(sender_to_offense_gate_dist_rank_relative_to_friends)
            features.append(sender_to_offense_gate_dist_rank_relative_to_opponents)
            features.append(sender_to_top_sideline_dist_rank_relative_to_friends)
            features.append(sender_to_top_sideline_dist_rank_relative_to_opponents)
            features.append(sender_team_closest_dist_to_offense_goal_line)
            features.append(sender_team_closest_dist_to_defense_goal_line_exclude_goalie)
            features.append(sender_team_closest_dist_to_top_sideline)
            features.append(sender_team_cloeset_dist_to_bottom_sideline)
            features.append(player_to_offense_gate_dist_rank_relative_to_friends)
            features.append(player_to_offense_gate_dist_rank_relative_to_opponents)
            features.append(player_to_top_sideline_dist_rank_relative_to_friends)
            features.append(player_to_top_sideline_dist_rank_relative_to_opponents)
            features.append(sender_team_median_dist_to_offense_goal_line)
            features.append(sender_team_median_dist_to_top_sideline)
            features.append(player_to_sender_dist_rank_among_friends)
            features.append(player_to_sender_dist_rank_among_opponents)
            features.append(min_pass_angle)
            features.append(num_dangerous_opponents_along_passing_line)
            features.append(player_closest_friend_to_sender_dist)
            features.append(player_closest_opponent_to_sender_dist)

            
            if get_features:
                yield features
            else:
                yield output_separator.join([str(feature) for feature in features])
        
    @staticmethod
    def get_header():
        features = [
            'pass_id',
            'line_num',
            'label',
            'sender_id',
            'player_id',
            'is_in_same_team',
            'time_start',
            #'duration',
            #'ball_speed',
            #'sender_x',
            #'sender_y',
            #'player_x',
            #'player_y',
            'distance',
            'min_opponent_dist_to_sender_player_line',
            'second_opponent_dist_to_sender_player_line',
            'third_opponent_dist_to_sender_player_line',
            'is_sender_in_back_field',
            'is_sender_in_middle_field',
            'is_sender_in_front_field',
            'is_player_in_back_field',
            'is_player_in_middle_field',
            'is_player_in_front_field',
            'is_sender_player_in_same_field',
            'sender_to_offense_gate_dist',
            'sender_to_defense_gate_dist',
            'player_to_offense_gate_dist',
            'player_to_defense_gate_dist',
            'norm_player_sender_x_diff',
            'is_player_in_offense_direction_relative_to_sender',
            'abs_y_diff',
            'is_start_of_game',
            'sender_to_center_distance',
            'player_to_center_distance',
            'is_player_in_center_circle',
            'is_sender_goal_keeper',
            'is_player_goal_keeper',
            'sender_closest_friend_dist',
            'sender_closest_3_friends_dist',
            'sender_closest_opponent_dist',
            'sender_closest_3_opponents_dist',
            'player_closest_friend_dist',
            'player_closest_3_friends_dist',
            'player_closest_opponent_dist',
            'player_closest_3_opponents_dist',
            'sender_to_offense_gate_dist_rank_relative_to_friends',
            'sender_to_offense_gate_dist_rank_relative_to_opponents',
            'sender_to_top_sideline_dist_rank_relative_to_friends',
            'sender_to_top_sideline_dist_rank_relative_to_opponents',
            'sender_team_closest_dist_to_offense_goal_line',
            'sender_team_closest_dist_to_defense_goal_line_exclude_goalie',
            'sender_team_closest_dist_to_top_sideline',
            'sender_team_cloeset_dist_to_bottom_sideline',
            'player_to_offense_gate_dist_rank_relative_to_friends',
            'player_to_offense_gate_dist_rank_relative_to_opponents',
            'player_to_top_sideline_dist_rank_relative_to_friends',
            'player_to_top_sideline_dist_rank_relative_to_opponents',
            'sender_team_median_dist_to_offense_goal_line',
            'sender_team_median_dist_to_top_sideline',
            'player_to_sender_dist_rank_among_friends',
            'player_to_sender_dist_rank_among_opponents',
            'min_pass_angle',
            'num_dangerous_opponents_along_passing_line',
            'player_closest_friend_to_sender_dist',
            'player_closest_opponent_to_sender_dist'
        ]
        return features
        
    def __get_min_opponent_dist_to_sender_player_line(self, player_id):
        if not self.__in_same_team(self.sender_id, player_id):
            return [-1] * 11
        opponents = {id: self.players[id] for id in self.players.keys() if not self.__in_same_team(player_id, id)}
        sender = self.players[self.sender_id]
        player = self.players[player_id]
        sender_to_player_vec = np.array([player.x - sender.x, player.y - sender.y])
        dummy_large_distance = 50000
        if np.linalg.norm(sender_to_player_vec) < 0.0001: # if sender to player distance is too small, then don't need to calc
            return [dummy_large_distance] * 11
        distances = []
        for id in opponents.keys():
            opponent = opponents[id]
            sender_to_opponent_vec = np.array([opponent.x - sender.x, opponent.y - sender.y])
            # refer to https://blog.csdn.net/tracing/article/details/46563383
            sender_to_projection_point_vec = \
              (np.dot(sender_to_player_vec, sender_to_opponent_vec) / np.linalg.norm(sender_to_player_vec)) * \
              (sender_to_player_vec / np.linalg.norm(sender_to_player_vec))
            projection_point_to_opponent_vec = sender_to_opponent_vec - sender_to_projection_point_vec
            distance = np.linalg.norm(projection_point_to_opponent_vec)
            projection_point_vec = np.array([sender.x, sender.y]) + sender_to_projection_point_vec
            if ((projection_point_vec[0] >= sender.x and projection_point_vec[0] <= player.x) \
              or (projection_point_vec[0] <= sender.x and projection_point_vec[0] >= player.x)):
                distances.append(distance)
        if len(distances) < 11:
            distances += [dummy_large_distance] * (11 - len(distances))
        distances.sort()
        return distances
        
    def __get_player_field(self, player_id, is_player_left_team):
        # divide the field into (back, middle, front), and represent it as (0, 1, 2)
        player = self.players[player_id]
        if math.fabs(player.x) <= 1750:
            return 1
        if (is_player_left_team and player.x < 0) or (not is_player_left_team and player.x > 0):
            return 0
        return 2
        
    def __normalized_player_to_sender_x_diff(self, player_id, is_sender_left_team):
        # set all offense direction to be positive and defense direction to be negative
        x_diff = self.players[player_id].x - self.players[self.sender_id].x
        return x_diff if is_sender_left_team else (-1 * x_diff)
        
    def __is_start_of_game(self):
        home_players = [id for id in self.players.keys() if id <= 14]
        away_players = [id for id in self.players.keys() if id > 14]
        is_home_left_team = self.__is_player_left_team(home_players[0])
        left_players = home_players if is_home_left_team else away_players
        right_players = away_players if is_home_left_team else home_players
        threshold = 200
        left_max_x = np.max([self.players[id].x for id in left_players])
        right_min_x = np.min([self.players[id].x for id in right_players])
        return 1 if left_max_x <= threshold and right_min_x >= -threshold else 0
        
    def __is_player_left_team(self, player_id):
        # or right team
        team_players_x = [self.players[id].x for id in self.players.keys() if self.__in_same_team(player_id, id)]
        oppo_players_x = [self.players[id].x for id in self.players.keys() if not self.__in_same_team(player_id, id)]
        return np.mean(team_players_x) < np.mean(oppo_players_x)
        
    def __player_to_gate_distance(self, player_id, is_player_left_team):
        player = self.players[player_id]
        offense_gate = [5250, 0] if is_player_left_team else [-5250, 0]
        defense_gate = [-5250, 0] if is_player_left_team else [5250, 0]
        player_to_offense_gate_dist = np.linalg.norm(np.array([player.x , player.y]) - np.array(offense_gate))
        player_to_defense_gate_dist = np.linalg.norm(np.array([player.x , player.y]) - np.array(defense_gate))
        return player_to_offense_gate_dist, player_to_defense_gate_dist

    def __player_to_goal_line_distance(self, player_id, is_player_left_team):
        player = self.players[player_id]
        offense_goal_line_x = 5250 if is_player_left_team else -5250
        defense_goal_line_x = -5250 if is_player_left_team else 5250
        player_to_offense_goal_line_dist = math.fabs(offense_goal_line_x - player.x)
        player_to_defense_goal_line_dist = math.fabs(player.x - defense_goal_line_x)
        return player_to_offense_goal_line_dist, player_to_defense_goal_line_dist
        
    def __distance_to_center(self, player_id):
        return np.sqrt(self.players[player_id].x ** 2 + self.players[player_id].y ** 2)

    def __is_goal_keeper(self, player_id, is_player_left_team):
        friends = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if candidate_id != player_id and self.__in_same_team(player_id, candidate_id)}
        friends_x = [self.players[friend].x for friend in friends]
        if is_player_left_team:
            min_x = np.min(friends_x)
            return 1 if self.players[player_id].x <= min_x else 0
        else:
            max_x = np.max(friends_x)
            return 1 if self.players[player_id].x >= max_x else 0
        
    def __in_same_team(self, player_id_1, player_id_2):
        return (player_id_1 - 14.5) * (player_id_2 - 14.5) > 0
        
    def __get_distance(self, player_id_1, player_id_2):
        # verify ids before calling
        return np.sqrt((self.players[player_id_1].x - self.players[player_id_2].x) ** 2 +
                       (self.players[player_id_1].y - self.players[player_id_2].y) ** 2)
        
    def __validate(self):
        if self.sender_id not in self.players or self.receiver_id not in self.players:
            return False
        self.sender = self.players[self.sender_id]
        self.receiver = self.players[self.receiver_id]
        return True
        
    def __get_sender_receiver_distance(self):
        # call __validate before this function
        return np.sqrt((self.sender.x - self.receiver.x) ** 2 + (self.sender.y - self.receiver.y) ** 2)

    def __calculate_sender_player_opponent_angle(self, sender_id, player_id, opponent_id):
        sender = self.players[sender_id]
        opponent = self.players[opponent_id]
        player = self.players[player_id]
        sender_to_player = np.array([player.x, player.y]) - np.array([sender.x, sender.y])
        sender_to_opponent = np.array([opponent.x, opponent.y]) - np.array([sender.x, sender.y])

        cosine_angle = np.dot(sender_to_player, sender_to_opponent) / \
                       (np.linalg.norm(sender_to_player) * np.linalg.norm(sender_to_opponent))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)


def pass_builder(pass_id, line_num, tokens):
    players_count = 28
    time_start = int(tokens[0])
    time_end = int(tokens[1])
    sender_id = int(tokens[2])
    receiver_id = int(tokens[3])
    a_pass = Pass(pass_id, line_num, time_start, time_end, sender_id, receiver_id)
    for i in range(players_count):
        x = tokens[4 + i]
        y = tokens[4 + players_count + i]
        if not x or not y:
            continue
        a_pass.add_player(i+1, float(x), float(y))
    return a_pass
        
def get_passes():
    first_line = True
    line_num = 0
    pass_id = 0
    passes = []
    for line in open('passes.csv'):
        line_num += 1
        if first_line:
            first_line = False
            continue
        tokens = line.split(',')
        if len(tokens) < 60: continue
        a_pass = pass_builder(pass_id, line_num, tokens)
        # validate
        if a_pass.sender_id not in a_pass.players: continue
        passes.append(a_pass)
        pass_id += 1
    return passes
        
def featurize():
    with open('train.tsv', 'w') as train_writer, open('val.tsv', 'w') as val_writer, open('test.tsv', 'w') as test_writer:
        has_checked_feature_count = False
        header_item_count = len(Pass.get_header())
        train_writer.write(output_separator.join(Pass.get_header()) + '\n')
        val_writer.write(output_separator.join(Pass.get_header()) + '\n')
        test_writer.write(output_separator.join(Pass.get_header()) + '\n')
        passes = get_passes()
        counter = len(passes)
        print('Total valid samples count: %d' % counter)
        random.seed(30)
        #random.shuffle(passes)
        train_counts = int(counter * 0.7)
        val_counts = int(counter * 0.1)
        print('Train count: %d, val count: %d, test count: %d' % (train_counts, val_counts, counter - train_counts - val_counts))
        for i in range(counter):
            if i % 1000 == 0: print(i)
            writer = train_writer if i <= train_counts else val_writer if i <= (train_counts+val_counts) else test_writer
            for feature in passes[i].features_generator():
                writer.write(feature + "\n")
                if not has_checked_feature_count:
                    feature_count = len(feature.split(output_separator))
                    assert feature_count == header_item_count, 'Feature count (%d) is not the same as header count (%d)' % (feature_count, header_item_count)
                    has_checked_feature_count = True
    with open('generated.cs', 'w') as writer:
        for feature in Pass.get_header():
            writer.write("double %s = features[i++];\n" % feature.strip())

def outputPassingFeaturesForAll():
    with open('passingfeatures.tsv', 'w') as feature_writer:
        feature_writer.write(Pass.get_header() + '\n')
        passes = get_passes()
        counter = len(passes)
        for i in range(counter):
            for feature in passes[i].features_generator():
                feature_writer.write(feature + '\n')

def featurize_svm():
    dir = r'./lightgbm/'
    with open(dir + 'rank.train', 'w') as train_writer, open(dir + 'rank.train.query', 'w') as train_query_writer, open(dir + 'rank.train.id', 'w') as train_id_writer,\
         open(dir + 'rank.val', 'w') as val_writer, open(dir + 'rank.val.query', 'w') as val_query_writer, open(dir + 'rank.val.id', 'w') as val_id_writer, \
         open(dir + 'rank.test', 'w') as test_writer, open(dir + 'rank.test.query', 'w') as test_query_writer, open(dir + 'rank.test.id', 'w') as test_id_writer:
        headers = Pass.get_header()
        passes = get_passes()
        counter = len(passes)
        random.seed(30)
        #random.shuffle(passes)
        train_counts = int(counter * 0.7)
        val_counts = int(counter * 0.1)
        feature_start_column = 5
        for i in range(counter):
            if i % 1000 == 0: print(i)
            writer = train_writer if i <= train_counts else val_writer if i <= (train_counts+val_counts) else test_writer
            query_writer = train_query_writer if i <= train_counts else val_query_writer if i <= (train_counts+val_counts) else test_query_writer
            id_writer = train_id_writer if i <= train_counts else val_id_writer if i <= (train_counts+val_counts) else test_id_writer
            query_counts = 0
            for features in passes[i].features_generator(get_features=True):
                query_counts += 1
                output_features = []
                output_features.append(str(features[2])) # label
                for j in range(feature_start_column, len(features)):
                    if features[j] != 0:
                        output_features.append('%d:%s' % (j - feature_start_column + 1, str(features[j])))
                        #output_features.append('%s:%s' % (headers[j], str(features[j])))
                writer.write('%s\n' % (" ".join(output_features)))
                id_writer.write('%d\t%d\t%d\n' % (features[0], features[1], features[4])) # pass_id, line_num, receiver_id
            query_writer.write('%d\n' % query_counts)
            
def read_lines(reader, num):
    lines = []
    for i in range(num):
        lines.append(reader.readline().strip())
    return lines
            
def lightgbm_pred_accuracy(label_file, query_file, predict_file, id_file, output_file):
    #os.popen('/mnt/c/source/github/LightGBM/lightgbm config=lightgbm/predict.conf')
    topN = 5
    counter = 0
    topn_correct_counters = [0] * topN
    feature_reader = open(label_file, 'r')
    predict_reader = open(predict_file, 'r')
    id_reader = open(id_file, 'r')
    writer = open(output_file, 'w')
    pass_lines = open(query_file).readlines()
    pass_lines = [int(count) for count in pass_lines if count.strip()]
    for count in pass_lines:
        labels = [int(line.split()[0]) for line in read_lines(feature_reader, count)]
        results = [float(line) for line in read_lines(predict_reader, count)]
        id_lines = read_lines(id_reader, count)
        pass_id = id_lines[0].split('\t')[0]
        line_num = id_lines[0].split('\t')[1]
        receiver_ids = [line.split('\t')[2] for line in id_lines]
        receiver = np.argmax(labels)
        top_predictions = np.argsort(results)[::-1]
        for i in range(topN):
            topn_predictions = top_predictions[:i+1]
            if receiver in topn_predictions:
                topn_correct_counters[i] += 1
        counter += 1
        ranked_receiver_ids = [receiver_ids[n] for n in top_predictions]
        writer.write('%s\t%s\t%s\n' % (pass_id, line_num, ",".join(ranked_receiver_ids)))
    topN_accuracies = [0] * topN
    for i in range(topN):
        topN_accuracies[i] = float(topn_correct_counters[i])/counter
        print("Top %d prediction accuracy: %d/%d = %f" % \
            (i+1, topn_correct_counters[i], counter, topN_accuracies[i]))
    writer.close()
    return topN_accuracies
    
def xgboost_pred_accuracy(label_file, query_file, predict_file, id_file, output_file):
    lightgbm_pred_accuracy(label_file, query_file, predict_file, id_file, output_file)

if USER == 'Zhiying':
    LIGHTGBM_EXEC = '/mnt/c/source/github/LightGBM/lightgbm'
elif USER == 'Heng':
    LIGHTGBM_EXEC = '/Users/hengli/Projects/mlsa18-pass-prediction/LightGBM/lightgbm'
def lightgbm_run(cmd):
    cwd = os.getcwd()
    os.chdir('./lightgbm')
    #os.popen(cmd)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print("Error while calling shell command: " + cmd)
    os.chdir(cwd)
    
def lightgbm_pipeline():
    print("Featurizing")
    featurize_svm()
    print("Train")
    lightgbm_run(LIGHTGBM_EXEC + ' config=train.conf > train.log')
    print("Predict")
    if USER == 'Zhiying':
        lightgbm_run('bash predict.sh')
    elif USER == 'Heng':
        lightgbm_run('bash predict_heng.sh')
    print("Train accuracies:")
    lightgbm_pred_accuracy('lightgbm/rank.train', 'lightgbm/rank.train.query', 'lightgbm/LightGBM_predict_train.txt', 'lightgbm/rank.train.id', 'lightgbm/rank.train.result')
    print("Validation accuracies:")
    lightgbm_pred_accuracy('lightgbm/rank.val', 'lightgbm/rank.val.query', 'lightgbm/LightGBM_predict_val.txt', 'lightgbm/rank.val.id', 'lightgbm/rank.val.result')
    print("Test accuracies:")
    lightgbm_pred_accuracy('lightgbm/rank.test', 'lightgbm/rank.test.query', 'lightgbm/LightGBM_predict_test.txt', 'lightgbm/rank.test.id', 'lightgbm/rank.test.result')
    lightgbm_feature_importance()
    
def lightgbm_feature_importance():
    model_file = 'lightgbm/LightGBM_model.txt'
    feature_importance = False
    headers = Pass.get_header()
    feature_start_column = 5
    for line in open(model_file):
        if "feature importance" in line:
            feature_importance = True
            continue
        if not feature_importance: continue
        if not line.startswith("Column_"):
            feature_importance = False
            break
        column = int(line.strip()[len("Column_"):line.find('=')])
        value = int(line.strip()[line.find('=')+1:])
        print('%s=%d' % (headers[column - 1 + feature_start_column], value))
    
def lightgbm_train():
    df_train = pd.read_csv('train.tsv', sep='\t')
    df_test = pd.read_csv('test.tsv', sep='\t')
    
    # https://github.com/Microsoft/LightGBM/blob/21487d8a28e53c63382d3ab8481b073b65176022/examples/python-guide/advanced_example.py
    
def update_config_file(params, input_config, output_config):
    writer = open(output_config, 'w')
    for line in open(input_config, 'r'):
        line = line.strip()
        for key in params.keys():
            if line.startswith(key):
                line = '%s = %s' % (key, str(params[key]))
                break
        writer.write(line + '\n')
    writer.close()
    
def lightgbm_train_test_with_param(params):
    update_config_file(params, 'lightgbm/train.default.config', 'lightgbm/train.tmp.config')
    print(str(params))
    print("Train")
    lightgbm_run(LIGHTGBM_EXEC + ' config=train.tmp.config >train.log')
    print("Predict")
    lightgbm_run('bash predict.sh')
    print("Train accuracies:")
    train_accuracies = lightgbm_pred_accuracy('lightgbm/rank.train', 'lightgbm/rank.train.query', 'lightgbm/LightGBM_predict_train.txt', 'lightgbm/rank.train.id', 'lightgbm/rank.train.result')
    print("Validation accuracies:")
    val_accuracies = lightgbm_pred_accuracy('lightgbm/rank.val', 'lightgbm/rank.val.query', 'lightgbm/LightGBM_predict_val.txt', 'lightgbm/rank.val.id', 'lightgbm/rank.val.result')
    print("Test accuracies:")
    lightgbm_pred_accuracy('lightgbm/rank.test', 'lightgbm/rank.test.query', 'lightgbm/LightGBM_predict_test.txt', 'lightgbm/rank.test.id', 'lightgbm/rank.test.result')
    return (train_accuracies, val_accuracies)
    
def params_generator(params_to_sweep):
    key_count = len(params_to_sweep.keys())
    assert key_count >= 1
    random_key = params_to_sweep.keys()[0]
    for value in params_to_sweep[random_key]:
        params = {random_key: value}
        if key_count == 1:
            yield params
        else:
            sub_params_to_sweep = params_to_sweep.copy()
            del sub_params_to_sweep[random_key]
            for dict in params_generator(sub_params_to_sweep):
                params.update(dict)
                yield params
        
def hyper_parameter_sweep():
    #params_to_sweep = {'learning_rate': [0.05, 0.1, 0.2], 'num_trees': [50, 100, 200, 500], 'num_leaves': [31, 63]}
    #params_to_sweep = {'learning_rate': [0.05, 0.07], 'num_trees': [200, 500, 1000], 'min_data_in_leaf': [50], 
    #    'feature_fraction': [0.5], 'bagging_fraction': [0.5], 'num_leaves': [15, 31, 63, 127]}
    params_count = 1
    for key in params_to_sweep.keys():
        params_count *= len(params_to_sweep[key])
    print('Totally %d parameters to sweep\n' % params_count)
    writer = open('tmp.tsv', 'w')
    results= {}
    top1_val_results = {}
    counter = 0
    for params in params_generator(params_to_sweep):
        print('\nParameter count: %d' % counter)
        counter += 1
        start_time = datetime.datetime.now()
        train_accuracies, val_accuracies = lightgbm_train_test_with_param(params)
        print('Finished in %f seconds' % (datetime.datetime.now() - start_time).total_seconds())
        writer.write('\n\n%s\nTrain accuracies: %s\nValid accuracies: %s\n' % (str(params), str(train_accuracies), str(val_accuracies)))
        writer.flush()
        top1_val_results[str(params)] = val_accuracies[0]
        results[str(params)] = (train_accuracies, val_accuracies)
    writer.write('\nSorted val accuracies:\n')
    for r in sorted(top1_val_results, key=top1_val_results.get, reverse=True):
        writer.write('%s: top1_val_acc: %f, top1_train_acc: %f\n' % (r, top1_val_results[r], results[r][0][0]))
    writer.close()
    
def avg_pred_results(readers, out_file):
    writer = open(out_file, 'w')
    for line in readers[0].readlines():
        test_results = []
        test_results.append(float(line))
        for i in range(1, len(readers)):
            test_results.append(float(readers[i].readline()))
        writer.write("%f\n" % np.mean(test_results))
    writer.close()
    
def model_ensemble():
    model_files = ['LightGBM_model_1.txt', 'LightGBM_model_2.txt']#, 'LightGBM_model_3.txt']
    output_train_files = []
    output_test_files = []
    reader_train_files = []
    reader_test_files = []
    for i, model_file in enumerate(model_files):
        output_train_file = 'LightGBM_predict_train_%d.txt' % i
        output_test_file = 'LightGBM_predict_test_%d.txt' % i
        params_train = {'input_model': model_file, 'output_result': output_train_file }
        params_test = {'input_model': model_file, 'output_result': output_test_file }
        update_config_file(params_train, 'lightgbm/predict_train.conf', 'lightgbm/predict_train.tmp.conf')
        lightgbm_run(LIGHTGBM_EXEC + ' config=predict_train.tmp.conf')
        update_config_file(params_test, 'lightgbm/predict_test.conf', 'lightgbm/predict_test.tmp.conf')
        lightgbm_run(LIGHTGBM_EXEC + ' config=predict_test.tmp.conf')
        reader_train_files.append(open('lightgbm/' + output_train_file, 'r'))
        reader_test_files.append(open('lightgbm/' + output_test_file, 'r'))
    # build feature file
    #gen_feature_file = 'lightgbm/model_ensemble_rank.train'
    avg_train_outfile = 'lightgbm/model_ensemble_avg_predict_train.txt'
    avg_test_outfile = 'lightgbm/model_ensemble_avg_predict_test.txt'
    avg_pred_results(reader_train_files, avg_train_outfile)
    avg_pred_results(reader_test_files, avg_test_outfile)
    print("Train accuracies:")
    lightgbm_pred_accuracy('lightgbm/rank.train', 'lightgbm/rank.train.query', avg_train_outfile, 'lightgbm/rank.train.id', 'lightgbm/rank.train.result')
    print("Test accuracies:")
    lightgbm_pred_accuracy('lightgbm/rank.test', 'lightgbm/rank.test.query', avg_test_outfile, 'lightgbm/rank.test.id', 'lightgbm/rank.test.result')
    
def lightgbm_python():
    X_train, y_train = load_svmlight_file('lightgbm/rank.train')
    X_val, y_val = load_svmlight_file('lightgbm/rank.val')
    X_test, y_test = load_svmlight_file('lightgbm/rank.test')
    q_train = np.loadtxt('lightgbm/rank.train.query')
    q_val = np.loadtxt('lightgbm/rank.val.query')
    gbm = lgb.LGBMRanker(learning_rate=0.05, n_estimators=500)
    gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_val, y_val)],
            eval_group=[q_val], eval_at=[1,3,5], verbose=False)
    #gbm.save_model('model.txt')
    y_pred = gbm.predict(X_test)
    np.savetxt('lightgbm/LightGBM_predict_test.txt', y_pred)
    #print(y_pred[:10])
    #print(y_pred.shape)
    
def remove_features(infile, outfile, allowed_features):
    writer = open(outfile, 'w')
    for line in open(infile):
        if not line.strip(): continue
        tokens = line.strip().split()
        output_features = []
        label = True
        for token in tokens:
            if label:
                output_features.append(token)
                label = False
                continue
            feature_id = int(token.split(':')[0])
            if feature_id in allowed_features:
                output_features.append(token)
        if len(output_features) < 2:
            output_features.append('%d:0' % list(allowed_features)[0])
        writer.write('%s\n' % ' '.join(output_features))
    writer.close()
    
def train_test_single_feature():
    headers = Pass.get_header()
    writer = open('tmp.tsv', 'w')
    results= {}
    top1_val_results = {}
    for i in range(1, 52):
        print('Feature %s (%d)' % (headers[i - 1 + 5], i))
        remove_features('lightgbm/rank.default.train', 'lightgbm/rank.train', set([i]))
        remove_features('lightgbm/rank.default.val', 'lightgbm/rank.val', set([i]))
        print("Train")
        lightgbm_run(LIGHTGBM_EXEC + ' config=train.conf > train.log')
        print("Predict")
        if USER == 'Zhiying':
            lightgbm_run('bash predict.sh')
        elif USER == 'Heng':
            lightgbm_run('bash predict_heng.sh')
        print("Train accuracies:")
        train_accuracies = lightgbm_pred_accuracy('lightgbm/rank.train', 'lightgbm/rank.train.query', 'lightgbm/LightGBM_predict_train.txt', 'lightgbm/rank.train.id', 'lightgbm/rank.train.result')
        print("Valid accuracies:")
        val_accuracies = lightgbm_pred_accuracy('lightgbm/rank.val', 'lightgbm/rank.val.query', 'lightgbm/LightGBM_predict_val.txt', 'lightgbm/rank.val.id', 'lightgbm/rank.val.result')
        top1_val_results[i] = val_accuracies[0]
        results[i] = (train_accuracies, val_accuracies)
    writer.write('\nSorted val accuracies:\n')
    for r in sorted(top1_val_results, key=top1_val_results.get, reverse=True):
        feature_name = headers[r - 1 + 5]
        writer.write('Feature %s (%d): top1_val_acc: %f, top1_train_acc: %f\n' % (feature_name, r, top1_val_results[r], results[r][0][0]))
    writer.close()
    
if __name__ == '__main__':
    #featurize()
    #featurize_svm()
    #lightgbm_pred_accuracy('lightgbm/rank.train', 'lightgbm/rank.train.query', 'lightgbm/LightGBM_predict_train.txt', 'lightgbm/rank.train.id', 'lightgbm/rank.train.result')
    #lightgbm_pred_accuracy('lightgbm/rank.test', 'lightgbm/rank.test.query', 'lightgbm/LightGBM_predict_test.txt', 'lightgbm/rank.test.id', 'lightgbm/rank.test.result')
    lightgbm_pipeline()
    #xgboost_pred_accuracy('xgboost/rank.test', 'xgboost/rank.test.group', 'xgboost/pred.txt', 'lightgbm/rank.test.id', 'xgboost/rank.test.result')
    #lightgbm_train_test_with_param({'num_leaves': 31, 'learning_rate': 0.05, 'min_data_in_leaf': 50, 'num_trees': 500, 'bagging_fraction': 0.5, 'feature_fraction': 0.5})
    #lightgbm_train_test_with_param({'learning_rate': 0.05, 'min_data_in_leaf': 500, 'num_trees': 500, 'bagging_fraction': 0.5, 'feature_fraction': 1})
    #lightgbm_train_test_with_param({'learning_rate': 0.07, 'min_data_in_leaf': 50, 'num_trees': 200})
    #hyper_parameter_sweep()
    #lightgbm_python()
    #model_ensemble()
    #train_test_single_feature()
    
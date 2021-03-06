from __future__ import print_function
import os
import random
import math
import numpy as np
#import lightgbm as lgbß

#root = r'C:\source\github\mlsa18-pass-prediction' if os.name == 'nt' else r'/mnt/c/source/github/mlsa18-pass-prediction'
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
        sender_friends = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if candidate_id != self.sender_id and self.__in_same_team(self.sender_id, candidate_id)}
        sender_opponents = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if not self.__in_same_team(self.sender_id, candidate_id)}
        sender_friends_distances = sorted([self.__get_distance(friend_id, self.sender_id) for friend_id in sender_friends.keys()])
        sender_opponent_distances = sorted([self.__get_distance(opponent_id, self.sender_id) for opponent_id in sender_opponents.keys()])
        sender_closest_friend_dist = sender_friends_distances[0]
        sender_closest_3_friends_dist = np.mean(sender_friends_distances[:3])
        sender_closest_opponent_dist = sender_opponent_distances[0]
        sender_closest_3_oppononents_dist = np.mean(sender_opponent_distances[:3])
        sender_field = self.__get_player_field(self.sender_id)
        is_sender_in_back_field = 1 if sender_field == 0 else 0
        is_sender_in_middle_field = 1 if sender_field == 1 else 0
        is_sender_in_front_field = 1 if sender_field == 2 else 0
    
        for player_id in self.players.keys():
            if player_id == self.sender_id: continue
            sender = self.players[self.sender_id]
            player = self.players[player_id]
            
            friends = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if candidate_id != player_id and self.__in_same_team(player_id, candidate_id)}
            opponents = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if not self.__in_same_team(player_id, candidate_id)}
            
            label = 1 if player_id == self.receiver_id else 0
            is_in_same_team = 1 if self.sender_id in friends.keys() else 0
            distance = self.__get_distance(self.sender_id, player_id)
            friend_distances = sorted([self.__get_distance(friend_id, player_id) for friend_id in friends.keys()])
            opponent_distances = sorted([self.__get_distance(opponent_id, player_id) for opponent_id in opponents.keys()])
            
            player_field = self.__get_player_field(player_id)
            is_player_in_back_field = 1 if player_field == 0 else 0
            is_player_in_middle_field = 1 if player_field == 1 else 0
            is_player_in_front_field = 1 if player_field == 2 else 0
            is_sender_player_in_same_field = 1 if sender_field == player_field else 0
            normalized_player_to_sender_x_diff = self.__normalized_player_to_sender_x_diff(player_id)
            
            features = []
            features.append(self.pass_id)
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
            features.append(self.__get_min_opponent_dist_to_sender_player_line(player_id))
            features.append(is_sender_in_back_field)
            features.append(is_sender_in_middle_field)
            features.append(is_sender_in_front_field)
            features.append(is_player_in_back_field)
            features.append(is_player_in_middle_field)
            features.append(is_player_in_front_field)
            features.append(is_sender_player_in_same_field)
            features.append(normalized_player_to_sender_x_diff)
            features.append(1 if normalized_player_to_sender_x_diff >= 0 else 0) # is_player_in_offense_direction_relative_to_sender
            features.append(self.__is_start_of_game())
            features.append(self.__distance_to_center(self.sender_id)) # sender_to_center_distance
            features.append(self.__distance_to_center(player_id)) # player_to_center_distance
            features.append(1 if self.__distance_to_center(player_id) <= 915 else 0) # is_player_in_center_circle
            features.append(self.__is_goal_keeper(self.sender_id)) # is_sender_goal_keeper
            features.append(self.__is_goal_keeper(player_id)) # is_player_goal_keeper
            features.append(sender_closest_friend_dist)
            features.append(sender_closest_3_friends_dist)
            features.append(sender_closest_opponent_dist)
            features.append(sender_closest_3_oppononents_dist)
            features.append(friend_distances[0]) # closest friend distance
            features.append(np.mean(friend_distances[:3])) # closest 3 friends avg distance
            features.append(opponent_distances[0]) # closest opponent distance
            features.append(np.mean(opponent_distances[:3])) # closest 3 opponent avg distance
            
            if get_features:
                yield features
            else:
                yield output_separator.join([str(feature) for feature in features])
        
    @staticmethod
    def get_header():
        features = [
            'pass_id',
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
            'is_sender_in_back_field',
            'is_sender_in_middle_field',
            'is_sender_in_front_field',
            'is_player_in_back_field',
            'is_player_in_middle_field',
            'is_player_in_front_field',
            'is_sender_player_in_same_field',
            'norm_player_sender_x_diff',
            'is_player_in_offense_direction_relative_to_sender',
            'is_start_of_game',
            'sender_to_center_distance',
            'player_to_center_distance',
            'is_player_in_center_circle',
            'is_sender_goal_keeper',
            'is_player_goal_keeper',
            'sender_closest_friend_dist',
            'sender_closest_3_friends_dist',
            'sender_closest_opponent_dist',
            'sender_closest_3_oppononents_dist',
            'player_closest_friend_dist',
            'player_closest_3_friends_dist',
            'player_closest_opponent_dist',
            'player_closest_3_oppononents_dist'
        ]
        return output_separator.join(features)
        
    def __get_min_opponent_dist_to_sender_player_line(self, player_id):
        if not self.__in_same_team(self.sender_id, player_id):
            return 0
        opponents = {id: self.players[id] for id in self.players.keys() if not self.__in_same_team(player_id, id)}
        sender = self.players[self.sender_id]
        player = self.players[player_id]
        sender_to_player_vec = np.array([player.x - sender.x, player.y - sender.y])
        min_distance = 50000 # dummy large distance
        if np.linalg.norm(sender_to_player_vec) < 0.0001: # if sender to player distance is too small, then don't need to calc
            return min_distance
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
              or (projection_point_vec[0] <= sender.x and projection_point_vec[0] >= player.x)) \
              and (distance < min_distance):
                min_distance = distance
        return min_distance
        
    def __get_player_field(self, player_id):
        # divide the field into (back, middle, front), and represent it as (0, 1, 2)
        player = self.players[player_id]
        if math.fabs(player.x) <= 1750:
            return 1
        is_player_left_team = self.__is_player_left_team(player_id)
        if (is_player_left_team and player.x < 0) or (not is_player_left_team and player.x > 0):
            return 0
        return 2
        
    def __normalized_player_to_sender_x_diff(self, player_id):
        # set all offense direction to be positive and defense direction to be negative
        is_sender_left_team = self.__is_player_left_team(self.sender_id)
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
        
    def __distance_to_center(self, player_id):
        return np.sqrt(self.players[player_id].x ** 2 + self.players[player_id].y ** 2)
        
    def __is_goal_keeper(self, player_id):
        friends = {candidate_id: self.players[candidate_id] for candidate_id in self.players.keys() if candidate_id != player_id and self.__in_same_team(player_id, candidate_id)}
        max_x = np.max([math.fabs(self.players[friend].x) for friend in friends])
        return 1 if math.fabs(self.players[player_id].x) >= max_x else 0
        
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
        train_writer.write(Pass.get_header() + '\n')
        val_writer.write(Pass.get_header() + '\n')
        test_writer.write(Pass.get_header() + '\n')
        passes = get_passes()
        counter = len(passes)
        print('Total valid samples count: %d' % counter)
        random.seed(30)
        #random.shuffle(passes)
        train_counts = int(counter * 0.8)
        val_counts = int(counter * 0.0)
        print('Train count: %d, val count: %d, test count: %d' % (train_counts, val_counts, counter - train_counts - val_counts))
        for i in range(counter):
            writer = train_writer if i <= train_counts else val_writer if i <= (train_counts+val_counts) else test_writer
            for feature in passes[i].features_generator():
                writer.write(feature + "\n")
    with open('generated.cs', 'w') as writer:
        for feature in Pass.get_header().split(output_separator):
            writer.write("double %s = features[i++];\n" % feature.strip())

def outputPassingFeatures():
    with open('passingfeatures.tsv', 'w') as feature_writer:
        feature_writer.write(Pass.get_header() + '\n')
        passes = get_passes()
        counter = len(passes)
        for i in range(counter):
            for feature in passes[i].features_generator():
                feature_writer.write(feature + '\n')

def featurize_svm():
    dir = r'./lightgbm/'
    with open(dir + 'rank.train', 'w') as train_writer, open(dir + 'rank.train.query', 'w') as train_query_writer, \
         open(dir + 'rank.val', 'w') as val_writer, open(dir + 'rank.val.query', 'w') as val_query_writer, \
         open(dir + 'rank.test', 'w') as test_writer, open(dir + 'rank.test.query', 'w') as test_query_writer:
        passes = get_passes()
        counter = len(passes)
        random.seed(30)
        #random.shuffle(passes)
        train_counts = int(counter * 0.7)
        val_counts = int(counter * 0.1)
        feature_start_column = 4
        for i in range(counter):
            if i % 1000 == 0: print(i)
            writer = train_writer if i <= train_counts else val_writer if i <= (train_counts+val_counts) else test_writer
            query_writer = train_query_writer if i <= train_counts else val_query_writer if i <= (train_counts+val_counts) else test_query_writer
            query_counts = 0
            for features in passes[i].features_generator(get_features=True):
                query_counts += 1
                output_features = []
                output_features.append(str(features[1])) # label
                for j in range(feature_start_column, len(features)):
                    if features[j] != 0:
                        output_features.append('%d:%s' % (j - feature_start_column + 1, str(features[j])))
                writer.write('%s\n' % (" ".join(output_features)))
            query_writer.write('%d\n' % query_counts)
            
def read_lines(reader, num):
    lines = []
    for i in range(num):
        lines.append(reader.readline().strip())
    return lines
            
def lightgbm_pred_accuracy(label_file, query_file, predict_file):
    #os.popen('/mnt/c/source/github/LightGBM/lightgbm config=lightgbm/predict.conf')
    topN = 5
    counter = 0
    topn_correct_counters = [0] * topN
    feature_reader = open(label_file, 'r')
    predict_reader = open(predict_file, 'r')
    pass_lines = open(query_file).readlines()
    pass_lines = [int(count) for count in pass_lines if count.strip()]
    for count in pass_lines:
        labels = [int(line.split()[0]) for line in read_lines(feature_reader, count)]
        results = [float(line) for line in read_lines(predict_reader, count)]
        receiver = np.argmax(labels)
        top_predictions = np.argsort(results)[::-1]
        for i in range(topN):
            topn_predictions = top_predictions[:i+1]
            if receiver in topn_predictions:
                topn_correct_counters[i] += 1
        counter += 1
    for i in range(topN):
        print("Top %d prediction accuracy: %d/%d = %f" % \
            (i+1, topn_correct_counters[i], counter, float(topn_correct_counters[i])/counter))
    
LIGHTGBM_EXEC = '/mnt/c/source/github/LightGBM/lightgbm'
def lightgbm_run(cmd):
    cwd = os.getcwd()
    os.chdir('./lightgbm')
    os.popen(cmd)
    os.chdir(cwd)
    
def lightgbm_pipeline():
    print("Featurizing")
    #featurize_svm()
    print("Train")
    lightgbm_run(LIGHTGBM_EXEC + ' config=train.conf >train.log')
    print("Predict")
    lightgbm_run('bash predict.sh')
    print("Train accuracies")
    lightgbm_pred_accuracy('lightgbm/rank.train', 'lightgbm/rank.train.query', 'lightgbm/LightGBM_predict_train.txt')
    print("Test accuracies")
    lightgbm_pred_accuracy('lightgbm/rank.test', 'lightgbm/rank.test.query', 'lightgbm/LightGBM_predict_test.txt')
    
if __name__ == '__main__':
    #featurize()
    outputPassingFeatures()
    #featurize_svm()
    #lightgbm_pred_accuracy('lightgbm/rank.train', 'lightgbm/rank.train.query', 'lightgbm/LightGBM_predict_train.txt')
    #lightgbm_pred_accuracy('lightgbm/rank.test', 'lightgbm/rank.test.query', 'lightgbm/LightGBM_predict_test.txt')
    #lightgbm_pipeline()
    

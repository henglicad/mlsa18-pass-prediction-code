# Feature List

## Game State Features

- **time_start**: time elapsed since the start of the half at the time the ball was passed by the sender.
- **is_start_of_game**: is it the start of any half of the game.

## Sender Position Features

- **is_sender_in_back_field**: is sender in the back field of the pitch.
- **is_sender_in_middle_field**: is sender in the middle field of the pitch.
- **is_sender_in_front_field**: is sender in the front field of the pitch.
- **sender_to_offense_gate_dist**: the distance between sender and sender's offense gate.
- **sender_to_defense_gate_dist**: the distance between sender and sender's defense gate.
- **sender_to_center_distance**: the distance between sender and the center of the pitch.
- **is_sender_goal_keeper**: is sender the goal keeper.
- **sender_closest_teammate_dist**: the distance between sender and his/her closest teammate.
- **sender_closest_3_teammates_dist**: the average distance between sender and his/her three closest teammates.
- **sender_closest_opponent_dist**: the distance between sender and his/her closest opponent.
- **sender_closest_3_opponents_dist**: the average distance between sender and his/her three closest opponents.

## Candidate Receiver Position Features

- **distance**: candidate receiver's distance to the sender.
- **is_in_same_team**: is candidate receiver in the same team as the sender.
- **is_receiver_in_back_field**: is candidate receiver in the back field of the pitch.
- **is_receiver_in_middle_field**: is candidate receiver in the middle field of the pitch.
- **is_receiver_in_front_field**: is candidate receiver in the front field of the pitch.
- **is_sender_receiver_in_same_field**: is sender and candidate receiver in the same field of the pitch.
- **receiver_to_offense_gate_dist**: the distance between the candidate receiver and his/her offense gate.
- **receiver_to_defense_gate_dist**: the distance between the candidate receiver and his/her defense gate.
- **norm_receiver_sender_x_diff**: normalized x-axis distance between candidate receiver and the sender with the sender's offense direction set as the positive x-axis direction. This feature is to capture whether the candidate receiver is in the offense direction or deffense direction compared to the sender.
- **is_receiver_in_offense_direction_relative_to_sender**: binarized value of norm_receiver_sender_x_diff.
- **abs_y_diff**: the absoluate value of the sender and candidate receiver's y-axis difference.
- **receiver_to_center_distance**: the distance between candidate receiver and the center of the pitch.
- **is_receiver_in_center_circle**: is candidate receiver in the center circle.
- **is_receiver_goal_keeper**: is candidate receiver the goal keeper.
- **receiver_closest_teammate_dist**: the distance between the candidate receiver and his/her closest teammate.
- **receiver_closest_3_teammates_dist**: the average distance between the candidate receiver and his/her three closest teammates.
- **receiver_closest_opponent_dist**: the distance between the candidate receiver and his/her closest opponent.
- **receiver_closest_3_opponents_dist**: the average distance between the candidate receiver and his/her three closest opponents.
- **receiver_closest_teammate_to_sender_dist**: the distance between the sender and the candidate receiver's closest teammate.
- **receiver_closest_opponent_to_sender_dist**: the distance between the sender and the candidate receiver's closest opponent. 

## Passing Path Features

- **min_opponent_dist_to_sender_receiver_line**: the minimum distance an opponent has to the passing line drawn between the sender and the candidate receiver. If the sender and the candidate receiver is not in the same team, this is set to -1. The assumption is that if there is an opponent too close to the passing line, then this receiver might not be a good candidate.
- **second_opponent_dist_to_sender_receiver_line**: similar to above, change from minimum to second minimum.
- **third_opponent_dist_to_sender_receiver_line**: similar to above, change from minimum to third minimum.
- **num_dangerous_opponents_along_passing_line**: number of dangerous opponents along the passing line. We define dangerous opponents as the following: an opponent whose distance to the sender and that to the candidate receiver are both less than the distance between the sender and the candiate receiver.
- **min_pass_angle**: We define the pass angle as the following: for a given set of (sender, receiver, opponent), the pass angle is the angle between the sender to receiver line and sender to opponent line. The *min_pass_angle* is the minimum pass angle among all the dangerous opponents along the sender to candidate receiver passing line.

## Team Position Features

- **sender_to_offense_gate_dist_rank_relative_to_teammates**:  the rank of the distance to the offense gate for the sender among all teammates. This depicts the relative position of the sender in the team to the offense gate. The offense gate is defined as the point (5250, 0) if sender is in the left team (i.e. the average x for all players in sender's team is less than 0), and the point (-5250, 0) if sender is in the right team.
- **sender_to_offense_gate_dist_rank_relative_to_opponents**: similar to above, the rank is for the sender among all opponents instead.
- **sender_to_top_sideline_dist_rank_relative_to_teammates**: the rank of the distance to the top sideline for the sender among all teammates. This depicts the relative position of the sender in the team to the top sideline. The top sideline is defined as the y=3400 line.
- **sender_to_top_sideline_dist_rank_relative_to_opponents**: similar to above, the rank is for the sender among all opponents instead.
- **receiver_to_offense_gate_dist_rank_relative_to_teammates**: similar to  *sender_to_offense_gate_dist_rank_relative_to_teammates*, the rank is for the candidate receiver among his/her teammates instead.
- **receiver_to_offense_gate_dist_rank_relative_to_opponents**: similar to *sender_to_offense_gate_dist_rank_relative_to_opponents*, the rank is for the candidate receiver among his/her opponents instead.
- **receiver_to_top_sideline_dist_rank_relative_to_teammates**: similar to *sender_to_top_sideline_dist_rank_relative_to_teammates*, the rank is for the candidate receiver among his/her teammates instead.
- **receiver_to_top_sideline_dist_rank_relative_to_opponents**: similar to *sender_to_top_sideline_dist_rank_relative_to_opponents*, the rank is for the candidate receiver among his/her opponents instead.
- **sender_team_closest_dist_to_offense_goal_line**: the closest distance to sender's offense goal line among all teammates of the sender. The offense goal line is defined as the x=5250 line if sender is in the left team, and the x=-5250 line if sender is in the right team.
- **sender_team_closest_dist_to_defense_goal_line_exclude_goalie**: the closest distance to the defense goal line among all teammates of the sender except for the goalie. The defense goal line is defined as the x=-5250 line if sender is in the left team, and the x=5250 line if sender is in the right team.
- **sender_team_closest_dist_to_top_sideline**: the closest distance to the top sideline among all teammates of the sender.
- **sender_team_cloeset_dist_to_bottom_sideline**: the closest distance to the bottom sideline among all teammates of the sender.
- **sender_team_median_dist_to_offense_goal_line**: the median distance to the offense goal line among all teammates of the sender.
- **sender_team_median_dist_to_top_sideline**: the median distance to the top sideline among all teammates of the sender.
- **receiver_to_sender_dist_rank_among_teammates**: the rank of the distance to the sender for the candidate receiver among all receiver's teammates.
- **receiver_to_sender_dist_rank_among_opponents**: the rank of the distance to the sender for the candidate receiver among all receiver's opponents.

## Feature Importance Output From LightGBM

**Feature** | **Importance**
:-- | :--
receiver_closest_opponent_dist | 715
norm_receiver_sender_x_diff | 660
abs_y_diff | 653
distance | 616
receiver_closest_opponent_to_sender_dist | 602
receiver_closest_3_opponents_dist | 558
receiver_to_center_distance | 521
receiver_closest_3_teammates_dist | 508
sender_closest_opponent_dist | 498
min_pass_angle | 467
receiver_to_defense_gate_dist | 457
min_opponent_dist_to_sender_receiver_line | 450
sender_to_center_distance | 445
receiver_closest_teammate_dist | 438
receiver_to_offense_gate_dist | 421
receiver_closest_teammate_to_sender_dist | 419
sender_closest_3_teammates_dist | 409
third_opponent_dist_to_sender_receiver_line | 377
sender_closest_3_opponents_dist | 371
sender_team_cloeset_dist_to_bottom_sideline | 366
second_opponent_dist_to_sender_receiver_line | 356
sender_team_closest_dist_to_top_sideline | 348
time_start | 340
receiver_to_offense_gate_dist_rank_relative_to_teammates | 328
sender_to_offense_gate_dist | 320
sender_team_median_dist_to_top_sideline | 317
sender_closest_teammate_dist | 316
sender_to_defense_gate_dist | 300
sender_team_closest_dist_to_offense_goal_line | 276
sender_team_median_dist_to_offense_goal_line | 254
sender_team_closest_dist_to_defense_goal_line_exclude_goalie | 245
receiver_to_offense_gate_dist_rank_relative_to_opponents | 238
receiver_to_sender_dist_rank_among_teammates | 179
receiver_to_top_sideline_dist_rank_relative_to_opponents | 166
receiver_to_sender_dist_rank_among_opponents | 159
receiver_to_top_sideline_dist_rank_relative_to_teammates | 157
sender_to_offense_gate_dist_rank_relative_to_opponents | 143
sender_to_offense_gate_dist_rank_relative_to_teammates | 127
num_dangerous_opponents_along_passing_line | 103
sender_to_top_sideline_dist_rank_relative_to_teammates | 101
sender_to_top_sideline_dist_rank_relative_to_opponents | 89
is_receiver_in_offense_direction_relative_to_sender | 47
is_in_same_team | 34
is_receiver_goal_keeper | 20
is_sender_receiver_in_same_field | 19
is_sender_goal_keeper | 17
is_receiver_in_middle_field | 13
is_receiver_in_center_circle | 10
is_sender_in_middle_field | 8
is_receiver_in_back_field | 6
is_receiver_in_front_field | 6
is_start_of_game | 3
is_sender_in_front_field | 2
is_sender_in_back_field | 2

## Per Feature Training Accuracy

**Feature** | **Top-1 test accuracy**
:-- | :--
receiver_closest_teammate_to_sender_dist | 0.245050
receiver_closest_opponent_to_sender_dist | 0.204620
third_opponent_dist_to_sender_receiver_line | 0.204620
second_opponent_dist_to_sender_receiver_line | 0.194719
min_opponent_dist_to_sender_receiver_line | 0.172442
distance | 0.139439
min_pass_angle | 0.132838
receiver_to_sender_dist_rank_among_frends | 0.122112
norm_receiver_sender_x_diff | 0.112211
receiver_to_top_sideline_dist_rank_relative_to_opponents | 0.106436
receiver_closest_opponent_dist | 0.098185
num_dangerous_opponents_along_passing_line | 0.096535
receiver_to_offense_gate_dist_rank_relative_to_opponents | 0.090759
receiver_to_sender_dist_rank_among_opponents | 0.089109
receiver_to_defense_gate_dist | 0.076733
abs_y_diff | 0.076733
receiver_to_center_distance | 0.075908
receiver_closest_3_opponents_dist | 0.074257
receiver_to_offense_gate_dist | 0.072607
is_sender_receiver_in_same_field | 0.068482
is_in_same_team | 0.067657
is_receiver_in_offense_direction_relative_to_sender | 0.062706
receiver_to_offense_gate_dist_rank_relative_to_teammates | 0.056931
receiver_closest_teammate_dist | 0.056106
receiver_closest_3_teammates_dist | 0.052805
is_receiver_in_front_field | 0.051980
receiver_to_top_sideline_dist_rank_relative_to_teammates | 0.046205
is_receiver_in_back_field | 0.040429
time_start | 0.038779
is_sender_in_back_field | 0.038779
is_sender_in_middle_field | 0.038779
is_sender_in_front_field | 0.038779
sender_to_offense_gate_dist | 0.038779
sender_to_defense_gate_dist | 0.038779
is_start_of_game | 0.038779
sender_to_center_distance | 0.038779
is_sender_goal_keeper | 0.038779
is_receiver_goal_keeper | 0.038779
sender_closest_teammate_dist | 0.038779
sender_closest_3_teammates_dist | 0.038779
sender_closest_opponent_dist | 0.038779
sender_closest_3_opponents_dist | 0.038779
sender_to_offense_gate_dist_rank_relative_to_teammates | 0.038779
sender_to_offense_gate_dist_rank_relative_to_opponents | 0.038779
sender_to_top_sideline_dist_rank_relative_to_teammates | 0.038779
sender_to_top_sideline_dist_rank_relative_to_opponents | 0.038779
sender_team_closest_dist_to_offense_goal_line | 0.038779
sender_team_closest_dist_to_defense_goal_line_exclude_goalie | 0.038779
sender_team_closest_dist_to_top_sideline | 0.038779
sender_team_cloeset_dist_to_bottom_sideline | 0.038779
sender_team_median_dist_to_offense_goal_line | 0.038779
sender_team_median_dist_to_top_sideline | 0.038779
is_receiver_in_center_circle | 0.035479
is_receiver_in_middle_field | 0.033828
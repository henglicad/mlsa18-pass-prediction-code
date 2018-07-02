# Feature List

## Game State Features

- **time_start**: time elapsed since the start of the half at the time the ball was passed by the sender.
- **is_start_of_game**: is it the start of a half of the game.

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

## Passing Path Features

- **min_opponent_dist_to_sender_receiver_line**: the minimum distance an opponent has to the passing line drawn between the sender and the candidate receiver. If the sender and the candidate receiver is not in the same team, this is set to -1. The assumption is that if there is an opponent too close to the passing line, then this receiver might not be a good candidate.
- **second_opponent_dist_to_sender_receiver_line**: similar to above, change from minimum to second minimum.
- **third_opponent_dist_to_sender_receiver_line**: similar to above, change from minimum to third minimum.

## Team Position Features

- **sender_to_offense_gate_dist_rank_relative_to_teammates**:  the rank of the distance to the offense gate for the sender among all teammates. This depicts the relative position of the sender in the team to the offense gate.
- **sender_to_offense_gate_dist_rank_relative_to_opponents**: similar to above, the rank is for the sender among all opponents instead.
- **sender_to_top_sideline_dist_rank_relative_to_teammates**: the rank of the distance to the top sideline for the sender among all teammates. This depicts the relative position of the sender in the team to the top sideline.
- **sender_to_top_sideline_dist_rank_relative_to_opponents**: similar to above, the rank is for the sender among all opponents instead.
- **receiver_to_offense_gate_dist_rank_relative_to_teammates**: similar to  *sender_to_offense_gate_dist_rank_relative_to_teammates*, the rank is for the candidate receiver among his/her teammates instead.
- **receiver_to_offense_gate_dist_rank_relative_to_opponents**: similar to *sender_to_offense_gate_dist_rank_relative_to_opponents*, the rank is for the candidate receiver among his/her opponents instead.
- **receiver_to_top_sideline_dist_rank_relative_to_teammates**: similar to *sender_to_top_sideline_dist_rank_relative_to_teammates*, the rank is for the candidate receiver among his/her teammates instead.
- **receiver_to_top_sideline_dist_rank_relative_to_opponents**: similar to *sender_to_top_sideline_dist_rank_relative_to_opponents*, the rank is for the candidate receiver among his/her opponents instead.
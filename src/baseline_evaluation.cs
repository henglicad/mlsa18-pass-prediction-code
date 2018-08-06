        public static void EvaluateModel()
        {
            //var inputFile = @"D:\source\github\mlsa18-pass-prediction\train.tsv";
            //var inputFile = @"D:\source\github\mlsa18-pass-prediction\test.tsv";
            var inputFile = @"D:\source\github\mlsa18-pass-prediction\all.tsv";
            //var inputFile = @"D:\source\github\mlsa18-pass-prediction\all_same_team.tsv";
            var passIdToPredictScores = new Dictionary<int, Dictionary<int, double>>();
            var passIdToReceiverId = new Dictionary<int, int>();
            //var isFirstLine = true;
            var isFirstLine = false;
            foreach (var line in File.ReadAllLines(inputFile))
            {
                if (isFirstLine)
                {
                    isFirstLine = false;
                    continue;
                }
                var tokens = line.Split('\t');
                if (tokens.Length < 31) continue;
                int i = 0;
                var pass_id = int.Parse(tokens[i++]);
                var line_num = int.Parse(tokens[i++]);
                var label = int.Parse(tokens[i++]);
                var sender_id = int.Parse(tokens[i++]);
                var player_id = int.Parse(tokens[i++]);
                var is_sender_in_back_field = int.Parse(tokens[11]);
                var is_sender_in_middle_field = int.Parse(tokens[12]);
                var is_sender_in_front_field = int.Parse(tokens[13]);
                var norm_player_sender_x_diff = double.Parse(tokens[22]);
                //if (is_sender_in_front_field != 1) continue;
                var features = new List<double>();
                for (; i<tokens.Length; i++)
                {
                    features.Add(double.Parse(tokens[i]));
                }
                //var prob = predict(features.ToArray());
                //var prob = predict_without_duration(features.ToArray());
                //var prob = predict_features_0531_2(features.ToArray());
                //var prob = predict_features_0601_1(features.ToArray());
                var is_in_same_team = int.Parse(tokens[5]);
                var distance = double.Parse(tokens[7]);
                if (distance <= 0)
                {
                    //Console.WriteLine();
                }
                System.Diagnostics.Debug.Assert(distance >= 0.0 && (distance == 0 || distance > 1.0));
                var prob = is_in_same_team * (norm_player_sender_x_diff >= 0 ? 10 : 0 + 1.0 / (distance + 1e-8)); // naive prediction based on distance
                //var prob = is_in_same_team * (1.0 / (distance + 1e-8));
                if (!passIdToPredictScores.ContainsKey(pass_id)) passIdToPredictScores.Add(pass_id, new Dictionary<int, double>());
                passIdToPredictScores[pass_id].Add(player_id, prob);
                if (label == 1) passIdToReceiverId.Add(pass_id, player_id);
            }

            int[] topNs = new int[] { 1, 2, 3, 4, 5};
            int[] topNCorrectCounters = new int[topNs.Length];
            double MRR_total = 0.0;
            var totalCount = passIdToReceiverId.Keys.Count;
            foreach (var passId in passIdToReceiverId.Keys)
            {
                var receiverId = passIdToReceiverId[passId];
                var scores = passIdToPredictScores[passId].OrderByDescending(x => x.Value).ToList();
                for (int j=0; j<topNs.Length; j++)
                {
                    var topPredicts = scores.Take(topNs[j]).Select(x => x.Key).ToList();
                    if (topPredicts.Contains(receiverId))
                    {
                        topNCorrectCounters[j]++;
                    }
                }
                for (int j=0; j<scores.Count; j++)
                {
                    if (scores[j].Key == receiverId || scores[j].Value <= 0.0)
                    {
                        MRR_total += 1.0 / (j + 1);
                        break;
                    }
                }
            }

            for (int j= 0; j < topNs.Length; j++)
            {
                Console.WriteLine($"Top {topNs[j]} prediction accuracy: {topNCorrectCounters[j]}/{totalCount} = {topNCorrectCounters[j]/(double)totalCount}");
            }
            Console.WriteLine($"MRR is: {MRR_total / totalCount}");
        }

        public static void Run()
        {
            EvaluateModel();
        }
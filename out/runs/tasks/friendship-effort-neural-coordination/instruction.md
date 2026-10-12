# Friendship Effort and Neural Coordination Task

You are given synthetic participant-level summaries from three-person joint force tasks with social-network and EEG-hyperscanning features. Each triad contains two close friends and one non-friend. Using train.csv, predict which test participants are non-friends based on effort allocation and neural coordination summaries.

Inputs:
- /app/environment/data/train.csv: labeled training participants. Columns: participant_id, triad_id, nonfriend, classroom_degree, mean_force, force_variability, alpha_erd, beta_erd, isc, phase_sync.
- /app/environment/data/test.csv: unlabeled test participants with the same columns except nonfriend.

Output:
Write /app/predictions.json as a JSON object with a predictions array. Each array item must contain participant_id and nonfriend_prob, where nonfriend_prob is a float between 0 and 1. Include every participant_id from test.csv.

Scoring:
Your score is the AUROC of nonfriend_prob against hidden nonfriend labels in test.csv. A perfect ranking scores 1.0; a random ranking scores about 0.5.

Constraints:
- Do not read or modify /tests or /logs/verifier/reward.json.
- Use only the provided environment data.
- Internet access is not required.

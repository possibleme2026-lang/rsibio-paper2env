# STP-EI mismatch detection in synthetic CA1 pattern trains

Implement the simplified STP/EI mismatch detector described in the paper. Input: /app/environment/data/inputs.json. It contains events_per_trial and patterns. patterns is a hex string; each event is 4 hex characters. The first 2 characters encode 8 excitatory (E) bouton activations and the next 2 encode 8 inhibitory (I) bouton activations. Bit b (0 = least significant bit) is active if 1. Events are grouped into trials of length events_per_trial; STP states reset at trial start.

For each trial, initialize each E and I bouton with x=1.0 and u=0.5. For each event, for each active bouton compute response r=u*x, then update x=x-r and u=u+0.5*(1-u). After processing all active boutons in the event, apply recovery to every bouton x=x+0.5*(1-x). Sum responses over E boutons and over I boutons. The mismatch score for that event is abs(E_sum - I_sum).

Write /app/predictions.json as a JSON object whose scores field is a list of finite floats, one per event in input order. The hidden verifier regenerates gold binary labels from the same STP/EI model and computes AUROC of your scores against those labels. Do not read /tests; no network access is available.

To apply the bias mitigation methods that are incorporated within this work:

# ILE
In recbole\model\general_recommender\bpr.py line 52:
In order, the five values are the lambdas of STD, ENT, Euclidean distance, KL-divergence, and MAD.
Make sure exactly one of these larger than 0 and all others are 0.

# IPS
In recbole\model\general_recommender\bpr.py line 53:
Set self.IPS to True.
In recbole\trainer\trainer.py line 170:
Set IPS_max to a value larger than 1. This is the maximum value of the re-mapping range.

# CP
In recbole\quick_start\quick_start.py line 157:
Set calibrate to True.
In recbole\evaluator\collector.py lines 173-174:
Set weight to the desired value of lambda (between 0-1), set self.topm to the desired size of the large lists.

# PUFR
In recbole\quick_start\quick_start.py line 157:
Set uncertainty to True.
In recbole\evaluator\collector.py line 234:
Set uncertainty_weight to the desired value of lambda, must be larger than 0.
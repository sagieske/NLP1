def calculate_metrics( correct, predictions, label):
	"""
	Calculates the recacll, precision and f1 score for the label
	source: https://stats.stackexchange.com/questions/51296/how-to-calculate-precision-and-recall-for-multiclass-classification-using-confus
	precision is the fraction of events where we correctly declared i out of all instances where the algorithm declared i. 
	Recall is the fraction of events where we correctly declared i out of all of the cases where the true of state of the world is i.
	"""
	t_pos = 0
	t_neg = 0
	f_pos = 0
	f_neg = 0
	for i in range(0, len(correct)):
		if correct[i] == label and predictions[i] == label:
			t_pos += 1
		elif correct[i] == label:
			f_neg += 1
		elif predictions[i] == label:
			f_pos += 1
		else:
			f_neg += 1
	relevant = t_pos + f_neg
	retrieved = t_pos + f_pos
	# Calculate recall
	if relevant == 0:
		recall = 0.0
	else:
		recall = t_pos /float(relevant)
	# Calculate precision
	if retrieved == 0:
		precision = 0.0
	else:
		precision = t_pos /float(retrieved)
	# Calculate f1
	if precision+recall == 0:
		f1 = 0.0
	else:
		f1 = 2 * (precision * recall )/float(precision+recall)
	return recall, precision, f1
				


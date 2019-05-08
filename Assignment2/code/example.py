from  lamdamart import LambdaMART
import numpy as np


def get_data(file_loc):
	with open(file_loc, 'r') as f:
		data = []
		i = 0
		for line in f:
			new_arr = []
			arr = line.split(' #')[0].split()
			''' Get the score and query id '''
			score = arr[0]
			q_id = arr[1].split(':')[1]
			new_arr.append(int(score))
			new_arr.append(int(q_id))
			arr = arr[2:]
			''' Extract each feature from the feature vector '''
			for el in arr:
				new_arr.append(float(el.split(':')[1]))
			data.append(new_arr)
			i += 1
			if i == 100:
				break;
	f.close()
	return np.array(data)

def main():
	training_data = get_data('code/train.txt')
	print(training_data)
	test_data = get_data('code/test.txt')

	model = LambdaMART(training_data=training_data, number_of_trees=1, learning_rate=0.1)
	model.fit()
	model.save('example_model')


	average_ndcg, predicted_scores = model.validate(test_data, 10)
	predicted_scores = model.predict(test_data[:,1:])
	print("Predicted Scores")
	print(predicted_scores)


main()
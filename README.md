# ICU Hypothesis Project

This is the repository for generating and testing complex hypotheses using the RETAIN model ([Choi et al. 2017](https://arxiv.org/abs/1608.05745)) and a custom attention-based LSTM generator.

## Run code
### Package requirements
- Python 3.x
- Tensorflow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

### Hardware requirement
The code was run using tigergpu. Training RETAIN locally is not recommended. Training the generator is feasible on the local machine.

### RETAIN Predictor
To train: ```python retain_train.py --num_codes=N --emb_size=M --path_data_train=/path/to/train/data.pkl --path_data_test=/path/to/test/data.pkl --path_target_train=/path/to/train/target.pkl --path_target_test=/path/to/test/target.pkl --directory=/output/directory```

### LSTM Generator


## Contributors
- Daniel Chae (Princeton Class of 2020), advised by Michael Guerzhoy
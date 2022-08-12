### Method for dataset imbalance ###
* CustomTrainer was used as a subclass to Trainer class, with CrossEntropy being assigned weights in compute_loss function.
* Weightage to each class was approximated according to the number of samples of that class in dataset.
* There wasn't any option to hyperparameter tune these class weights, so manual values of class weights were assigned and results were then seen to reach some conclusion on value of weights to each class.
* Resultingly there was little improvement in predictions of undersampled classes.  

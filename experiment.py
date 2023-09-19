from .svm import SupportVectorMachine
def accuracy_measure(y_true, y_prec):
    """
    Accurcy which is tp + tn /(tp + fp + tn + fn)
    :param y_true: gorund truth labels
    :param y_prec: predictions
    :return: a scalar float between 0 and 1
    """

def train_model(list_of_hyperparamters: list[dict[str,float]]) -> tuple(dict[str, float], float, float):
    """
    After loading and preprocesisng the dataset, this function will iterate over all hyperparamters that are given train
     a model on the training set  and evaluate the models on the validation set. After loading and preprocesisng the dataset.
     the hyperparamters will be given in the following format [{"num_epochs": 100, "C": 0.1" , "learning_rate": 0.01}...]

    :param list_of_hyperparamters:  a list of dictionary that contains the hyperparamter in the following format [{"num_epochs": 100, "C": 0.1" , "learning_rate": 0.01}...]
    :return: best hyper parameters, the best accuracy on the val set and the accurcy of the model on the
    test set
    """


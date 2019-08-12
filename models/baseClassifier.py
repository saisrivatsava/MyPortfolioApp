from models import GenericCode
from models import classification_models
from models import regression_models

class BaseClassifer:

    def __init__(self):
        self.genericCode = GenericCode.GenericCode()
        self.classifier = classification_models.Classifier()
        self.regressor = regression_models.Regressor()

    def trainWithoutTestFile(self, data, target, features_to_exclude, model_type):
        if model_type == "Classification":
            return self.classifier.train(data, target, features_to_exclude)
        else:
            return self.regressor.train(data, target, features_to_exclude)


    def trainWithTestFile(self,train_data,test_data,target,col_names,run_name,features_to_exclude,model_type):

        if model_type == "Classification":
            return self.classifier.trainWithTestData(train_data,test_data,target,col_names,run_name,features_to_exclude)
        else:
            return self.regressor.trainWithTestData(train_data,test_data,target,col_names,run_name,features_to_exclude)

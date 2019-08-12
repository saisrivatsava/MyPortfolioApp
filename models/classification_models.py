import collections


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from models import GenericCode


class Classifier:


    def __init__(self):
        self.genericCode = GenericCode.GenericCode()

    classifiersMap = {
        "KNeighborsClassifier": KNeighborsClassifier(3),
        "SVC": SVC(kernel="rbf", C=0.025, probability=True),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "BaggingClassifier": BaggingClassifier(),
        "SGDClassifier": SGDClassifier(),
        "RidgeClassifier": RidgeClassifier()
    }

    scoresMap = {}
    argsMap = {}

    def train(self, data, target, features_to_exclude_list):
        self.raw_data, self.X_raw, self.y_raw, self.X_train, self.X_test, self.y_train, self.y_test, runDetails = self.genericCode.loadAndSplitData(
            data, target, features_to_exclude_list)
        pipesList = self.genericCode.getNumericAndCatFeatures(self.X_train)
        preprocessor = self.genericCode.createPreprocessTransformer(pipesList)
        sorted_scores_map = self.fitModelsMap(preprocessor)
        return (runDetails, sorted_scores_map)

    def trainWithTestData(
            self,
            train_data,
            test_data,
            target,
            col_names,
            run_name,
            features_to_exclude_list):
        self.raw_data, self.X_raw, self.y_raw, self.X_train, self.X_test, self.y_train, self.y_test, runDetails = self.genericCode.loadAndSplitData(
            train_data, target, features_to_exclude_list)
        pipesList = self.genericCode.getNumericAndCatFeatures(self.X_train)
        preprocessor = self.genericCode.createPreprocessTransformer(pipesList)
        sorted_scores_map = self.fitModelsMap(preprocessor)
        sorted_scores_keys = sorted_scores_map.keys()
        models_args_list = []
        for modelName in sorted_scores_keys:
            models_args_list.append(self.argsMap[modelName])

        highest_scored_model = self.getHighestScoreModel(sorted_scores_map)
        highest_scored_model_name = list(sorted_scores_map.keys())[0]
        cross_val_score_mean = self.getCrossValScore(
            preprocessor, highest_scored_model)

        predicted_target_series = self.predictTarget(
            test_data, preprocessor, highest_scored_model)

        default_cols_Df = test_data[col_names.split(",")]
        predicted_Df = pd.DataFrame({target: predicted_target_series})
        outputDf = pd.concat([default_cols_Df, predicted_Df], axis=1)
        output_folder = "outputFiles/"
        output_f_name = self.genericCode.saveFile(
            run_name, output_folder, outputDf)
        return (
            runDetails,
            sorted_scores_map,
            models_args_list,
            output_f_name,
            cross_val_score_mean,
            highest_scored_model_name)

    def getCrossValScore(self, preprocessor, model):
        pipe = Pipeline(
            steps=[
                ('preprocessor', preprocessor), ('classifier', model)])
        cv_scores = cross_val_score(pipe, self.X_raw, self.y_raw, cv=10)
        mean_cv_score = sum(cv_scores) / len(cv_scores)
        return mean_cv_score

    def predictTarget(self, test_data, preprocessor, highest_scored_model):
        pipe = Pipeline(
            steps=[
                ('preprocessor',
                 preprocessor),
                ('classifier',
                 highest_scored_model)])
        pipe.fit(self.X_raw, self.y_raw)

        predictions = pipe.predict(test_data)
        return predictions

    def getHighestScoreModel(self, sorted_map):
        modelsList = list(sorted_map.keys())
        model = self.classifiersMap[modelsList[0]]
        return model

    def fitModelsMap(self, preprocessor):
        scoresMap = {}

        for name, cmodel in self.classifiersMap.items():
            pipe = Pipeline(
                steps=[
                    ('preprocessor', preprocessor), ('classifier', cmodel)])
            pipe.fit(self.X_train, self.y_train)
            score = pipe.score(self.X_test, self.y_test)
            # score_arr = cross_val_score(pipe, self.X_raw, self.y_raw, cv=10)
            # score = score_arr.mean()
            classifierArgs = str(cmodel.get_params())
            scoresMap.update({name: score})
            self.argsMap.update({name: classifierArgs})

        sorted_scores = sorted(
            scoresMap.items(),
            key=lambda kv: kv[1],
            reverse=True)
        sorted_scores_map = collections.OrderedDict(sorted_scores)
        return sorted_scores_map

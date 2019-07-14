import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import collections

class Classifier:
    X_train = ""
    X_test =""
    y_train =""
    y_test =""
    raw_data = ""
    X_raw=""
    y_raw=""

    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    # NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]

    classifiersMap = {
    "KNeighborsClassifier":  KNeighborsClassifier(3),
    "SVC":    SVC(kernel="rbf", C=0.025, probability=True),
    "DecisionTreeClassifier":    DecisionTreeClassifier(),
    "RandomForestClassifier":    RandomForestClassifier(),
    "RandomForestClassifier":    AdaBoostClassifier(),
    "GradientBoostingClassifier":    GradientBoostingClassifier()}

    scoresMap={}
    argsMap={}

    def train(self, data, target):
        self.loadAndSplitData(data, target)
        featuresList = self.getNumericAndCatFeatures(target)
        preprocessor = self.createPreprocessTransformer(featuresList)
        res  = self.fitModels(preprocessor)
        runDetails = self.getRunDetails()
        return (runDetails,res)

    def trainWithTestData(self, train_data, test_data, target, col_names, runName):
        self.loadAndSplitData(train_data, target)
        # print("X_train "+str(len(self.X_train)))
        # print("X_test "+str(len(self.X_test)))
        featuresList = self.getNumericAndCatFeatures(target)
        preprocessor = self.createPreprocessTransformer(featuresList)
        sorted_scores_map  = self.fitModelsMap(preprocessor)
        runDetails = self.getRunDetails()
        sorted_scores_keys = sorted_scores_map.keys()
        models_args_list = []
        for modelName in sorted_scores_keys:
            models_args_list.append(self.argsMap[modelName])

        highest_scored_model = self.getHighestScoreModel(sorted_scores_map)
        predicted_target = self.predictTarget(test_data,preprocessor, highest_scored_model)

        default_cols_Df= test_data[col_names.split(",")]
        predicted_Df = pd.DataFrame({target : predicted_target})
        outputDf = pd.concat([default_cols_Df, predicted_Df], axis=1)
        output_f_name=self.savePredictedFile(outputDf, runName)
        return (runDetails,sorted_scores_map, models_args_list, output_f_name)


    def savePredictedFile(self, outputDf, runName):
        output_file_name = "OutputFiles/"+runName+"Prediction.csv"
        outputDf.to_csv(output_file_name, index = False)
        # filename = secure_filename(file.filename)
        return output_file_name



    def predictTarget(self, test_data,preprocessor, highest_scored_model):
        pipe2 = Pipeline(steps=[('preprocessor', preprocessor),('classifier', highest_scored_model)])
        model = highest_scored_model
        pipe2.fit(self.X_raw, self.y_raw)

        predictions = pipe2.predict(test_data)
        return predictions


    def getRunDetails(self):
        resList = []
        resList.append(len(self.raw_data))
        resList.append(len(self.X_train))
        resList.append(len(self.X_test))

        return resList

    def getHighestScoreModel(self, sorted_map):
        modelsList = list(sorted_map.keys())
        model = self.classifiersMap[modelsList[0]]
        return model


    def loadAndSplitData(self, data, target):
        self.raw_data = data
        X = self.raw_data.drop(target, axis=1)
        y = self.raw_data[target]
        self.X_raw = X
        self.y_raw = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def createPreprocessTransformer(self,featureList ):
        cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ('scalar', StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, featureList[0]),
        ('cat', cat_pipe, featureList[1])])

        return preprocessor


    def getNumericAndCatFeatures(self, target):
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X_train.select_dtypes(include=['object']).columns
        return [numeric_features,categorical_features]


    def fitModels(self, preprocessor):
        result = []
        i = 1
        for classifier in self.classifiers:
            pipe = Pipeline(steps=[('preprocessor', preprocessor),('classifier', classifier)])
            pipe.fit(self.X_train, self.y_train)
            classifierName = type(classifier).__name__
            classifierArgs = str(classifier.get_params())
            # print(str(classifierArgs))
            resStr = str(i)+"|"
            i+=1
            resStr+=classifierName+"|"
            resStr+=classifierArgs+"|"
            resStr+=str(pipe.score(self.X_test, self.y_test))
            result.append(resStr)
            # print(classifier)
            # print("model score: %.3f" % pipe.score(self.X_test, self.y_test))
        return result

    def fitModelsMap(self, preprocessor):
        scoresMap = {}

        for name, cmodel in self.classifiersMap.items():
            # print(name)
            # print(model)
            pipe = Pipeline(steps=[('preprocessor', preprocessor),('classifier', cmodel)])
            pipe.fit(self.X_train, self.y_train)
            # classifierName = type(classifier).__name__
            score = pipe.score(self.X_test, self.y_test)
            classifierArgs = str(cmodel.get_params())
            scoresMap.update({name:score })
            self.argsMap.update({name:classifierArgs})

        sorted_scores = sorted(scoresMap.items(), key=lambda kv: kv[1], reverse=True)
        sorted_scores_map = collections.OrderedDict(sorted_scores)
        return sorted_scores_map
            # # print(str(classifierArgs))
            # resStr = str(i)+"|"
            # i+=1
            # resStr+=classifierName+"|"
            # resStr+=classifierArgs+"|"
            # resStr+=str(pipe.score(self.X_test, self.y_test))
            # result.append(resStr)
            # # print(classifier)
            # print("model score: %.3f" % pipe.score(self.X_test, self.y_test))
        # return result
    def __del__(self):
        print("Cleared Object")

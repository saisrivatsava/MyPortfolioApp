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

class Classifier:
    X_train = ""
    X_test =""
    y_train =""
    y_test =""
    raw_data = ""

    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    # NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]

    def train(self, data, target):
        self.loadAndSplitData(data, target)

        featuresList = self.getNumericAndCatFeatures(target)
        preprocessor = self.createPreprocessTransformer(featuresList)
        res  = self.fitModels(preprocessor)
        runDetails = self.getRunDetails()
        return (runDetails,res)


    def getRunDetails(self):
        resList = []
        resList.append(len(self.raw_data))
        resList.append(len(self.X_train))
        resList.append(len(self.X_test))

        return resList



    def loadAndSplitData(self, data, target):
        self.raw_data = data
        X = self.raw_data.drop(target, axis=1)
        y = self.raw_data[target]
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

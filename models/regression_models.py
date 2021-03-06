from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import  cross_val_score
from models import GenericCode
from sklearn.pipeline import Pipeline
import collections
import pandas as pd

# import xgboost as xgb
# import lightgbm as lgb


class Regressor:


    def __init__(self):
        self.genericCode = GenericCode.GenericCode()

    regressorsMap = {
    "LinearRegression":LinearRegression(),
    "DecisionTreeRegressor":DecisionTreeRegressor(),
    "Lasso":Lasso(alpha =0.005),#alpha =0.0005, random_state=1),
    "ElasticNet":ElasticNet(l1_ratio=.9),#alpha=0.0005, l1_ratio=.9, random_state=3),
    "KernelRidge":KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
    "GradientBoostingRegressor":GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5),

    "RandomForestRegressor":RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=None, oob_score=False, random_state=None,
           verbose=0, warm_start=False)}
    # "XGBRegressor":xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
    #                          learning_rate=0.05, max_depth=3,
    #                          min_child_weight=1.7817, n_estimators=2200,
    #                          reg_alpha=0.4640, reg_lambda=0.8571,
    #                          subsample=0.5213, silent=1,
    #                          random_state =7, nthread = -1)}

    scoresMap = {}
    argsMap = {}

    def train(self, data, target, features_to_exclude_list):
        self.raw_data, self.X_raw, self.y_raw, self.X_train, self.X_test, self.y_train, self.y_test, runDetails = self.genericCode.loadAndSplitData(
            data, target, features_to_exclude_list)
        pipesList = self.genericCode.getNumericAndCatFeatures(self.X_train)
        preprocessor = self.genericCode.createPreprocessTransformer(pipesList)
        sorted_scores_map = self.fitModelsMap(preprocessor)
        return (runDetails, sorted_scores_map)


    def fitModelsMap(self, preprocessor):
        scoresMap = {}

        for name, cmodel in self.regressorsMap.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', cmodel)])
            pipe.fit(self.X_train, self.y_train)
            # y_pred = pipe.predict(self.X_test)
            # score = mean_absolute_error(self.y_test, y_pred)
            score = pipe.score(self.X_test, self.y_test)
            classifierArgs = str(cmodel.get_params())
            scoresMap.update({name: score})
            self.argsMap.update({name: classifierArgs})

        sorted_scores = sorted(
            scoresMap.items(),
            key=lambda kv: kv[1],
            reverse=True)
        sorted_scores_map = collections.OrderedDict(sorted_scores)
        return sorted_scores_map

    def trainWithTestData(self,train_data,test_data,target,col_names,run_name,features_to_exclude_list):
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

        cross_val_score_mean = self.getCrossValScore(preprocessor, highest_scored_model)

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


    def getCrossValScore(self, preprocessor, model):
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        cv_scores = cross_val_score(pipe, self.X_raw, self.y_raw,scoring='r2', cv=10)
        mean_cv_score = cv_scores.mean()
        return mean_cv_score

    def getHighestScoreModel(self, sorted_map):
        modelsList = list(sorted_map.keys())
        model = self.regressorsMap[modelsList[0]]
        return model
# cross_val_score(tree_reg, housing_prepared, housing_labels,
# scoring="neg_mean_squared_error", cv=10)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class GenericCode:

    def getNumericAndCatFeatures(self, X_train):
        numeric_features = X_train.select_dtypes(
            include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(
            include=['object']).columns
        return [numeric_features, categorical_features]

    def createPreprocessTransformer(self, pipesList):
        cat_pipe = Pipeline(
            steps=[
                ('imputer', SimpleImputer(
                    strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(
                        handle_unknown='ignore'))])
        num_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy="median")),
            ('scalar', StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipe, pipesList[0]),
            ('cat', cat_pipe, pipesList[1])])

        return preprocessor

    def loadAndSplitData(self, data, target, features_to_exclude_list):
        dataDetails = []
        if features_to_exclude_list != "none":
            raw_data = data.drop(features_to_exclude_list.split(","), axis=1)
        else:
            raw_data = data
        X = raw_data.drop(target, axis=1)
        y = raw_data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        dataDetails.append(len(raw_data))
        dataDetails.append(len(X_train))
        dataDetails.append(len(X_test))

        return raw_data, X, y, X_train, X_test, y_train, y_test, dataDetails

    def saveFile(self, runName, output_folder, outputDf):
        output_file_name = runName + "Prediction.csv"
        outputDf.to_csv(output_folder + output_file_name, index=False)
        # filename = secure_filename(file.filename)
        return output_file_name

    def getHighestScoreModel(self, sorted_map):
        modelsList = list(sorted_map.keys())
        model =sorted_map[modelsList[0]]
        return model

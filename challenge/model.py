import json
import os

import joblib

import numpy as np
import pandas as pd

from typing import Tuple, Union, List
from datetime import datetime
from pathlib import Path

from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).parent.resolve()


class DelayModel:

    def __init__(self):
        self._model: LogisticRegression | None = None  # Model should be saved in this attribute.
        self._model_version: str = '1.0'
        self._model_path: Path = CURRENT_DIR.parent / "models" / f"v{self._model_version}"
        self.top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    @staticmethod
    def get_period_day(date: str) -> str:
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()

        if morning_min < date_time < morning_max:
            return 'mañana'
        elif afternoon_min < date_time < afternoon_max:
            return 'tarde'
        elif (evening_min < date_time < evening_max) or (night_min < date_time < night_max):
            return 'noche'

    @staticmethod
    def is_high_season(fecha: str) -> int:
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

        if (
                (range1_min <= fecha <= range1_max) or
                (range2_min <= fecha <= range2_max) or
                (range3_min <= fecha <= range3_max) or
                (range4_min <= fecha <= range4_max)
        ):
            return 1
        else:
            return 0

    @staticmethod
    def get_min_diff(data: pd.DataFrame) -> float:
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    @staticmethod
    def is_delay(data: pd.DataFrame, threshold_in_minutes: int = 15) -> ndarray:
        return np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

    def get_train_features(self, data: pd.DataFrame, target_column: str = None) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        # Never used, but in notebook.
        # training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state=111)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )

        # Getting just the top 10 features
        features = features.reindex(columns=self.top_10_features, fill_value=0)

        if target_column:
            target = data[[target_column]]
            return features, target

        return features

    def store_model(self, x_test, y_test):
        self._model_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, self._model_path / "lr_model.pkl")

        y_pred = self._model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        metadata = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": self._model_version,
            "features": self.top_10_features,
            "accuracy": {
                "class_0": report['0'],
                "class_1": report['1'],
                "accuracy": report['accuracy'],
            },
        }

        with open(self._model_path / "lr_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Getting period of day
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)

        # Getting if Schedule Date is at high season
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)

        # Getting minutes difference of scheduled time and operation time of the flight
        data['min_diff'] = data.apply(self.get_min_diff, axis=1)

        # If target column is set, setting delay label if delay > 15 minutes
        if target_column:
            data[target_column] = self.is_delay(data)
            data[target_column].value_counts()

        # Getting features and target dataframes
        return self.get_train_features(data, target_column)

    def fit(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Splitting data for train and test
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        # Getting 1s and 0s count for balance training
        n_y0 = (y_train == 0).sum()
        n_y1 = (y_train == 1).sum()

        # Training Logistic Regression with top 10' features and balance weights
        self._model = LogisticRegression(class_weight={1: n_y0 / len(y_train), 0: n_y1 / len(y_train)})
        self._model.fit(x_train, y_train)

        self.store_model(x_test, y_test)

    def predict(
            self,
            features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        
        return

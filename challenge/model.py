import numpy as np
import pandas as pd

from typing import Tuple, Union, List
from datetime import datetime

from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier


class DelayModel:

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

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

    @staticmethod
    def get_train_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Never used, but in notebook.
        training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state=111)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )

        target = data['delay']

        return features, target

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

        # Setting delay label if delay > 15 minutes
        data['delay'] = self.is_delay(data)

        # Getting features and target
        features, target = self.get_train_features(data)

        # Splitting train test data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

        # If there's target column, we can get the feature importance's.
        if target_column:
            X = data.drop([target_column])
            y = data[target_column]

            xgb = XGBClassifier(random_state=1, learning_rate=0.01)
            xgb.fit(X, y)

        XGBClassifier

        return

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
        return

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

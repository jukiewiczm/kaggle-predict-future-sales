from xgboost import XGBRegressor
from modeling.models.base_model import Model


class XGBWrapper(Model):
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)

    def fit(self, train, y_train, valid_set_tuple=None):
        if valid_set_tuple:
            # Only one valid set is used
            valid_set_tuple = [valid_set_tuple]

        self.model.fit(train, y_train, eval_set=valid_set_tuple)

    def transform(self, test):
        return self.model.predict(test)

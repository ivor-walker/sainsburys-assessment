"""
BaseModel - provides generic public methods to interact with the encapsulated model
"""
class BaseModel:
    def __init__(self,
        model
    ):
        self._model = model
        self._model_name = model.__class__.__name__
    
    def get_model_name(self):
        return self._model_name

    def load(self, path: str):
        self._model.load(path)
    
    def train(self, data):
        self._model.train(data)

    def save(self, path: str):
        self._model.save(path)

    def diagnose(self, data):
        return self._model.diagnose(data)

    def predict(self, data):
        return self._model.predict(data)

    def eval(self, data):
        # TODO
        pass

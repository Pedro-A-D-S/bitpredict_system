import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON


bit_predict_runner = bentoml.tensorflow.get(
    'bit-predict-final'
).to_runner()

svc = bentoml.Service(
    name='bit-predict-service',
    runners=[bit_predict_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def forecast(input_series: np.ndarray) -> np.ndarray:
    result = bit_predict_runner.run(input_series)

    return result

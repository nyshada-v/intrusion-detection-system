import numpy as np
from inference import predict_flow

dummy_flow = np.random.rand(78)

result = predict_flow(dummy_flow)

print(result)
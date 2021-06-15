from src.cloud import score_mnist
import torch
import json
import numpy as np
import pytest


@pytest.mark.parametrize("shape",[(1,1,28,28),(5,1,28,28)])
@pytest.mark.parametrize("dtype",[float,int,bool])
def test_init_and_run(shape,dtype):
    score_mnist.init()
    x_new = np.ones(shape,dtype=dtype)
    # Convert the array to a serializable list in a JSON document
    input_json = json.dumps({"data": x_new.tolist()})

    res = json.loads(score_mnist.run(input_json))

    assert len(res) == shape[0]
    assert isinstance(res[0], dict)
    assert tuple(res[0].keys()) == tuple(str(x) for x in range(10))
    assert all(isinstance(x,float) for x in res[0].values())

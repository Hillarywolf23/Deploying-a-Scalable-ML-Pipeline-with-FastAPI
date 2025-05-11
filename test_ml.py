import pytest
import numpy as np 
from ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
     Test that the train_model function returns a RandomForestClassifier instance.

    """
    X = np.random.rand(10,5)
    y = np.random.randint(0, 2, size=10)
    model = train_model(X,y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics_output():
    """
    Test that compute_model_metrics returns precision, recall, and fbeta as floats.

    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

# TODO: implement the third test. Change the function name and input as needed
def test_inference_output_shape():
    """
    Test that the inference function returns predictions with the correct shape.

    """
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, size=20)
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == (20,)

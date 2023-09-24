import os
import pytest
import pickle
import tensorflow as tf
from ..models import LR, SVM, NaiveBayes, LSTM
from new_samples import negative,positive
# Define the path to the "models" folder in your root repository
MODELS_FOLDER = "./models"

# Define a list of model files
MODEL_FILES = [
    "LR.pkl",
    "SVM.pkl",
    "NB.pkl",
    "LSTM.h5"
]

@pytest.mark.parametrize("model_file", MODEL_FILES)
def test_sentiment_models(model_file):
    model_path = os.path.join(MODELS_FOLDER, model_file)
    
    if model_file.endswith(".pkl"):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
    elif model_file.endswith(".h5"):
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError("Unsupported model file format")

    # Replace with actual test data
    test_data = [positive, negative]

    # Perform predictions
    predictions = model.predict(test_data)

    # Assert that predictions are of the expected shape or type
    assert isinstance(predictions, (list, np.ndarray))
    assert len(predictions) == len(test_data)

if __name__ == "__main__":
    pytest.main()

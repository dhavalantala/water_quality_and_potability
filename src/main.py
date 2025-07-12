from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import WaterPotabilityDataModel

app = FastAPI(
    title="Water Potability Prediction API",
    description="API for predicting water potability using a trained model.",
    version="1.0.0"
)

with open("../model.pkl", "rb") as model_file:
    model = pickle.load(model_file) 

@app.get("/")
def read_root():
    return {"message": "Welcome to the Water Potability Prediction API!"}

@app.post("/predict/")
def predict_potability(water: WaterPotabilityDataModel):
    """
    Predict water potability based on input features.
    
    :param data: Dictionary containing feature values.
    :return: Prediction result.
    """
    sample = pd.DataFrame({
        'ph' : [water.ph],
        'Hardness' : [water.Hardness],
        'Solids' : [water.Solids],
        'Chloramines' : [water.Chloramines],
        'Sulfate' : [water.Sulfate],
        'Conductivity' : [water.Conductivity],
        'Organic_carbon' : [water.Organic_carbon],
        'Trihalomethanes' : [water.Trihalomethanes],
        'Turbidity' : [water.Turbidity]
    })

    predicted_value = model.predict(sample)
    
    if predicted_value == 1:
        return"The water is Potable."
    else:
        return "The water is not Potable."



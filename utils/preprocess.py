import numpy as np

def preprocess_input(data, scaler):
    # Feature engineering
    rooms_per_household = data["AveRooms"] / data["HouseAge"]
    population_per_household = data["Population"] / data["AveOccup"]

    features = np.array([[
        data["MedInc"],
        data["HouseAge"],
        data["AveRooms"],
        data["AveBedrms"],
        data["Population"],
        data["AveOccup"],
        data["Latitude"],
        data["Longitude"],
        rooms_per_household,
        population_per_household
    ]])

    return scaler.transform(features)
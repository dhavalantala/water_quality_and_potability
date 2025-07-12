import json 
import requests

url = "https://render-p4pi.onrender.com/predict/"

x_new = dict(
    ph =  5.5,
    Hardness = 180,
    Solids = 32876,
    Chloramines = 5.9,
    Sulfate = 277.60,
    Conductivity = 496.36,
    Organic_carbon = 12.78,
    Trihalomethanes = 66.26,
    Turbidity = 5.14
)

x_new_jason = json.dumps(x_new)  

response = requests.post(url, data = x_new_jason)

print("Response from the server:", response.text)
print("Status code:", response.status_code)
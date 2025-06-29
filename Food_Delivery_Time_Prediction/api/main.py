# app/main.py

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.inference import predict_delivery_time

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class InputData(BaseModel):
    delivery_person_id: str
    delivery_person_age: str
    delivery_person_ratings: str
    restaurant_latitude: float
    restaurant_longitude: float
    delivery_location_latitude: float
    delivery_location_longitude: float
    order_date: str
    time_orderd: str
    time_order_picked: str
    weatherconditions: str
    road_traffic_density: str
    vehicle_condition: int
    type_of_order: str
    type_of_vehicle: str
    multiple_deliveries: str
    festival: str
    city: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict")
def predict(data: InputData):
    try:
        data_dict = data.model_dump()
        prediction = predict_delivery_time(data_dict)
        return {"predicted_delivery_time": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    delivery_person_id: str = Form(...),
    delivery_person_age: str = Form(...),
    delivery_person_ratings: str = Form(...),
    restaurant_latitude: float = Form(...),
    restaurant_longitude: float = Form(...),
    delivery_location_latitude: float = Form(...),
    delivery_location_longitude: float = Form(...),
    order_date: str = Form(...),
    time_orderd: str = Form(...),
    time_order_picked: str = Form(...),
    weatherconditions: str = Form(...),
    road_traffic_density: str = Form(...),
    vehicle_condition: int = Form(...),
    type_of_order: str = Form(...),
    type_of_vehicle: str = Form(...),
    multiple_deliveries: str = Form(...),
    festival: str = Form(...),
    city: str = Form(...)
):
    input_data = {
        "delivery_person_id": delivery_person_id,
        "delivery_person_age": delivery_person_age,
        "delivery_person_ratings": delivery_person_ratings,
        "restaurant_latitude": restaurant_latitude,
        "restaurant_longitude": restaurant_longitude,
        "delivery_location_latitude": delivery_location_latitude,
        "delivery_location_longitude": delivery_location_longitude,
        "order_date": order_date,
        "time_orderd": time_orderd,
        "time_order_picked": time_order_picked,
        "weatherconditions": weatherconditions,
        "road_traffic_density": road_traffic_density,
        "vehicle_condition": vehicle_condition,
        "type_of_order": type_of_order,
        "type_of_vehicle": type_of_vehicle,
        "multiple_deliveries": multiple_deliveries,
        "festival": festival,
        "city": city
    }

    try:
        prediction = predict_delivery_time(input_data)
        return templates.TemplateResponse("index.html", {"request": request, "result": prediction})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {e}"})


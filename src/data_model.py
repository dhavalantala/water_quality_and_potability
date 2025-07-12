from pydantic import BaseModel, Field

class WaterPotabilityDataModel(BaseModel):
    ph: float = Field(..., description="pH level of the water")
    Hardness: float = Field(..., description="Hardness of the water")
    Solids: float = Field(..., description="Total dissolved solids in the water")
    Chloramines: float = Field(..., description="Chloramines level in the water")
    Sulfate: float = Field(..., description="Sulfate level in the water")
    Conductivity: float = Field(..., description="Conductivity of the water")
    Organic_carbon: float = Field(..., description="Organic carbon content in the water")
    Trihalomethanes: float = Field(..., description="Trihalomethanes level in the water")
    Turbidity: float = Field(..., description="Turbidity of the water")


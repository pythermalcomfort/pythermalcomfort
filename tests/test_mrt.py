# tests/test_mrt.py
from pythermalcomfort.environment.mrt import mrt
from pythermalcomfort.classes_input import ForthPowerInputs, AreaWeightedInputs

def test_forth_power_si():
    result = mrt(ForthPowerInputs(surface_temps=[27.0, 17.0, 37.0], angle_factors=[0.4, 0.3, 0.3]))
    print(f"ForthPower SI: {result}")

def test_forth_power_ip():
    result = mrt(ForthPowerInputs(surface_temps=[80.6, 62.6, 98.6], angle_factors=[0.4, 0.3, 0.3], units="IP"))
    print(f"ForthPower IP: {result}")

def test_area_weighted_si():
    result = mrt(AreaWeightedInputs(surface_temps=[27.0, 17.0, 37.0], surface_areas=[5, 5.5, 20]))
    print(f"AreaWeighted SI: {result}")

if __name__ == "__main__":
    test_forth_power_si()
    test_forth_power_ip()
    test_area_weighted_si()
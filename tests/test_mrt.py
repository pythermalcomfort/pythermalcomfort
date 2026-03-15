# tests/test_mrt.py
from pythermalcomfort.environment.mrt import calculate_mrt
from pythermalcomfort.classes_input import ForthPowerInputs, NoGeometryInputs

def test_forth_power_si():
    result = calculate_mrt(ForthPowerInputs(surface_temps=[27.0, 17.0, 37.0], angle_factors=[0.4, 0.3, 0.3]))
    print(f"ForthPower SI: {result}")

def test_forth_power_ip():
    result = calculate_mrt(ForthPowerInputs(surface_temps=[80.6, 62.6, 98.6], angle_factors=[0.4, 0.3, 0.3], units="IP"))
    print(f"ForthPower IP: {result}")

def test_no_geometry_si():
    result = calculate_mrt(NoGeometryInputs(tdb=25, tr=30, asw=0.6))
    print(f"NoGeometry SI: {result}")

if __name__ == "__main__":
    test_forth_power_si()
    test_forth_power_ip()
    test_no_geometry_si()
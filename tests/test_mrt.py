# tests/test_mrt.py
from pythermalcomfort.environment.sky_emissivity import sky_emissivity
from pythermalcomfort.environment.sky_temperature import sky_temperature
from pythermalcomfort.environment.mrt import mrt
from pythermalcomfort.classes_input import ForthPowerInputs, AreaWeightedInputs, RayMenInputs, SkyEmissivityBruntInputs, SkyTemperatureInputs
from pythermalcomfort.environment.sky_emissivity import apply_dilley_correction

def test_forth_power_si():
    result = mrt(ForthPowerInputs(surface_temps=[27.0, 17.0, 37.0], angle_factors=[0.4, 0.3, 0.3]))
    print(f"ForthPower SI: {result}")

def test_forth_power_ip():
    result = mrt(ForthPowerInputs(surface_temps=[80.6, 62.6, 98.6], angle_factors=[0.4, 0.3, 0.3], units="IP"))
    print(f"ForthPower IP: {result}")

def test_area_weighted_si():
    result = mrt(AreaWeightedInputs(surface_temps=[27.0, 17.0, 37.0], surface_areas=[5, 5.5, 20]))
    print(f"AreaWeighted SI: {result}")

def test_ray_men_basic():
    """Test using the formal RayMenInputs structure."""
    
    # Zenith was 30, so Altitude is 90 - 30 = 60
    params = RayMenInputs(
        svf=1.0,           # Full sky view
        dni=800,           # Strong sun
        dhi=100,
        sol_altitude=60.0, # Sun 60 degrees above horizon
        t_surr=25.0,       # Warm surfaces
        albedo_ground=0.2
    )
    
    result = mrt(params)
    
    print(f"test_ray_men_basic SI: {result}")

def test_full_workflow_example() -> None:
    """Example of a user bridging Sky Emissivity -> Sky Temp -> MRT."""
    
    # 1. Weather Inputs
    t_air = 28.0      
    t_dew = 12.0      
    dni = 750         
    dhi = 150         
    sol_altitude = 60          
    
    # 2. Emissivity Step
    brunt_params = SkyEmissivityBruntInputs(tdp=t_dew)
    raw_eps = sky_emissivity(brunt_params)
    eps_val = apply_dilley_correction(raw_eps).eps_sky

    # 3. Sky Temperature Step
    # Converts Tdb and Eps to the effective blackbody temperature
    sky_temp_params = SkyTemperatureInputs(tdb=t_air, eps_sky=eps_val)
    t_sky_c = sky_temperature(sky_temp_params).t_sky

    raymen_params = RayMenInputs(
        svf=0.7,
        dni=dni,
        dhi=dhi,
        sol_altitude=sol_altitude, 
        t_surr=t_sky_c,
        albedo_ground=0.25,
        alpha_k=0.7,
        emissivity=0.97
    )

    result = mrt(raymen_params)
    
    print(f"test_full_workflow_example SI: {result}")

if __name__ == "__main__":
    test_forth_power_si()
    test_forth_power_ip()
    test_area_weighted_si()
    test_ray_men_basic()
    test_full_workflow_example()
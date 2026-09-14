# !pip install pythermalcomfort

from pythermalcomfort.models import pmv_ppd_iso, utci

# Calculate PMV and PPD using ISO 7730 standard
result = pmv_ppd_iso(
    tdb=25,  # Dry Bulb Air Temperature in $^\circ$C
    tr=25,  # Mean Radiant Temperature in $^\circ$C
    vr=0.1,  # Relative Air Speed in m/s
    rh=50,  # Relative Humidity in %
    met=1.4,  # Metabolic Rate in met
    clo=0.5,  # Dynamic Clothing Insulation in clo
    model="7730-2005",  # Year of the ISO standard
)
print(result)
# --------------------
#        PMVPPD
# --------------------
# pmv        : 0.41
# ppd        : 8.5
# tsv        : Neutral

# Calculate PMV and PPD using ISO 7730 standard in IP units
result = pmv_ppd_iso(
    tdb=77,  # Dry Bulb Air Temperature in $^\circ$F
    tr=77,  # Mean Radiant Temperature in $^\circ$F
    vr=0.1,  # Relative Air Speed in fps
    rh=50,  # Relative Humidity in %
    met=1.4,  # Metabolic Rate in met
    clo=0.5,  # Dynamic Clothing Insulation in clo
    model="7730-2005",  # Year of the ISO standard
    units="IP",  # Use IP units (Fahrenheit, fps, etc.)
)
print(result)
# --------------------
#        PMVPPD
# --------------------
# pmv        : 0.45
# ppd        : 9.2
# tsv        : Neutral

# Calculate UTCI using a list as input (vectorized calculation)
utci_value = utci(tdb=[30, 32], tr=30, v=0.5, rh=50)
print(utci_value)
# ----------------------------------------------------------------
#                               UTCI
# ----------------------------------------------------------------
# utci            : [30.40, 32.30]
# stress_category : [`moderate heat stress', `strong heat stress']

# print only the UTCI values
print(utci_value.utci)
# [30.4 32.3]

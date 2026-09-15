from pprint import pprint

from pythermalcomfort.environment import running_mean_outdoor_temperature
from pythermalcomfort.models import adaptive_ashrae

result = adaptive_ashrae(tdb=25, tr=25, t_running_mean=23, v=0.3)

pprint(result)

print(result.acceptability_80)  # or use result["acceptability_80"]

result = adaptive_ashrae(tdb=77, tr=77, t_running_mean=73.5, v=1, units="IP")

pprint(result)

print(result["acceptability_80"])

rmt_value = running_mean_outdoor_temperature([29, 28, 30, 29, 28, 30, 27], alpha=0.9)

result = adaptive_ashrae(tdb=25, tr=25, t_running_mean=rmt_value, v=0.3)
pprint(result)

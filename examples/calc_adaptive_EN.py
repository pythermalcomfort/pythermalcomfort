from pprint import pprint

from pythermalcomfort.environment import running_mean_outdoor_temperature
from pythermalcomfort.models import adaptive_en

result = adaptive_en(tdb=25, tr=25, t_running_mean=24, v=0.1)

pprint(result)

result = adaptive_en(tdb=22.5, tr=22.5, t_running_mean=24, v=0.1)

pprint(result)

comf_tmp = result["tmp_cmf"]

result = adaptive_en(tdb=72.5, tr=72.5, t_running_mean=75, v=0.1, units="IP")
pprint(result)

rmt_value = running_mean_outdoor_temperature([29, 28, 30, 29, 28, 30, 27], alpha=0.9)

result = adaptive_en(tdb=25, tr=25, t_running_mean=rmt_value, v=0.3)
pprint(result)

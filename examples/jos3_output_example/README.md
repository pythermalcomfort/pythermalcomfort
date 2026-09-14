# JOS-3 reference datasets

Digitised human-subject data used by `examples/calc_jos3.py` to compare JOS-3
predictions against measurements, and by
`examples/manuscript-v4/jos3_transient_validation.py`.

| File | Contents |
|---|---|
| `human_subject_experiment_dataset_Stolwijk1966.csv` | 441 rows across 9 exposure conditions (`A-FIG.4`–`A-FIG.7`, `B-FIG.1`–`B-FIG.5`): time, condition, rectal temperature, oesophageal temperature, mean skin temperature, metabolic rate, evaporative heat loss |
| `human_subject_experiment_dataset_Werner1980.csv` | 7 rows: local skin temperatures across 14 body sites at a range of operative temperatures |

## Provenance

J.A.J. Stolwijk, J.D. Hardy, "Partitional calorimetric studies of responses of
man to thermal transients", *J. Appl. Physiol.* 21(3) (1966) 967–977,
<https://doi.org/10.1152/jappl.1966.21.3.967>, and the companion paper
J.D. Hardy, J.A.J. Stolwijk, "Partitional calorimetric studies of man during
exposures to thermal transients", *J. Appl. Physiol.* 21(6) (1966) 1799–1806,
<https://doi.org/10.1152/jappl.1966.21.6.1799>. The two are commonly cited
together as "Stolwijk and Hardy, 1966".

The `Werner1980` sheet comes from the same dataset shipped with the original
JOS-3 release and is retained under the name used there.

## Reading them

Pass `float_precision="round_trip"`:

```python
pd.read_csv(path, header=0, float_precision="round_trip")
```

pandas' default C float parser can be one ULP out, which showed up as a
7.1e-15 discrepancy on one value per sheet. With `round_trip` the parsed
values are bit-identical to the spreadsheet these were exported from.

## History

These started life as two sheets in a single
`human_subject_experiment_dataset.xlsx`. Reading it required `openpyxl`, which
is not a dependency of this package and is not installed in the test
environments, so the examples failed for anyone running them as documented.

The sheets were exported verbatim, with no rounding, reordering, or renaming,
and verified to compare equal to the spreadsheet at `rtol=0, atol=0` including
dtypes. The spreadsheet and its one-off export script were then removed, since
keeping a second copy of the same numbers only invites the two to drift apart.

Both remain in git history if the original is ever needed:

```bash
git log --all --diff-filter=D -- '*human_subject_experiment_dataset.xlsx'
git show <commit>^:examples/jos3_output_example/human_subject_experiment_dataset.xlsx > recovered.xlsx
```

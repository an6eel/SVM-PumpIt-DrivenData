# SVM Pump it DrivenData Competition

## Preproccessing steps

### First submission

- Removed ["num_private", "recorded_by", "payment_type", "quantity_group", "waterpoint_type_group",
                  "management_group", "extraction_type_group", "extraction_type_class", "scheme_management"]

- One hot encoder for `funder` and `installer` variables

- Discretize categorical variables with `num_bins=20`. Use `num_chars=4` initial chars to group categorical data

- Replaced `NA` values in categorical variables with `unknown`

- Replaced `NA` values in logical variables (`permit` and `public_meeting`) with the most frequent element

- Normalize numerical variables

**SCORE --> 0.7842**

### Second submission

- Removed binarization for `funder` and `installer` variables.

### Best submission

- Changed `max_bins` value to 25 o 30

**SCORE --> 0.7966**

## Extra preprocessing

- Replace invalid data (ex: 0 for `construction_year`) with `median`
- Apply PCA with `n_components=10-20`

**SCORE --> 0.7952**






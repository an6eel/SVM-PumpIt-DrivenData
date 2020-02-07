# SVM Tanzanian-water DrivenData Competition

## Preproccessing steps

### First submission

- Removed ["num_private", "recorded_by", "payment_type", "quantity_group", "waterpoint_type_group",
                  "management_group", "extraction_type_group", "extraction_type_class", "scheme_management"]

- Binarize `funder` and `installer` variables

- Discretize categorical variables with `num_bins=20`. Use `num_chars=4` initial chars to group categorical data

- Replaced `NA` values in categorical variables with `unknown`

- Replaced `NA` values in logical variables (`permit` and `public_meeting`) with the most frequent element

- Normalize numerical variables

**SCORE --> 0.7842**

### Second submission

- Removed binarization for `funder` and `installer` variables.

**SCORE --> 0.7952**




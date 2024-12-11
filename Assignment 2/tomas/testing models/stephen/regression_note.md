# SVR 

## Preprocessing
### Missing Value Handle

I tried two methods:
1. fill missing value (999) with median value
2. drop entire row if missing value exist

### Outlier handle
1. replace with mean


Result:

config 1:
missing_val - fill_missing_values_with_median
outlier - replace_with_median
Mean Abs Error = 22.42

config 2:
missing_val - drop
outlier - replace_with_median
Mean Abs Error = 22.43


## next steps
- try different outlier handle methods
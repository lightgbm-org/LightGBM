LambdaRank Example
==================

Here is an example for LightGBM to run LambdaRank task.

***You must follow the [installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)
for the following commands to work. The `lightgbm` binary must be built and available at the root of this project.***

Training
--------

Run the following command in this folder:

```bash
"../../lightgbm" config=train.conf
```

Prediction
----------

You should finish training first.

Run the following command in this folder:

```bash
"../../lightgbm" config=predict.conf
```

Data Format
-----------

To learn more about the query format used in this example, check out the
[query data format](https://lightgbm.readthedocs.io/en/latest/Parameters.html#query-data).

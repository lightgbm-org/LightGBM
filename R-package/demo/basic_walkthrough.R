library(lightgbm)

# We load in the agaricus dataset
# In this example, we are aiming to predict whether a mushroom is edible
data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

# The loaded data is stored in sparseMatrix, and label is a numeric vector in {0,1}
class(train$label)
class(train$data)

# Set parameters for model training
train_params <- list(
    num_leaves = 4L
    , learning_rate = 1.0
    , objective = "binary"
    , nthread = 2L
)

#--------------------Basic Training using lightgbm----------------
# This is the basic usage of lightgbm you can put matrix in data field
# Note: we are putting in sparse matrix here, lightgbm naturally handles sparse input
# Use sparse matrix when your feature is sparse (e.g. when you are using one-hot encoding vector)
print("Training lightgbm with sparseMatrix")
bst <- lightgbm(
    data = train$data
    , params = train_params
    , label = train$label
    , nrounds = 2L
)

# Alternatively, you can put in dense matrix, i.e. basic R-matrix
print("Training lightgbm with Matrix")
bst <- lightgbm(
    data = as.matrix(train$data)
    , params = train_params
    , label = train$label
    , nrounds = 2L
)

# You can also put in lgb.Dataset object, which stores label, data and other meta datas needed for advanced features
print("Training lightgbm with lgb.Dataset")
dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
)
bst <- lightgbm(
    data = dtrain
    , params = train_params
    , nrounds = 2L
)

# Verbose = 0,1,2
print("Train lightgbm with verbose 0, no message")
bst <- lightgbm(
    data = dtrain
    , params = train_params
    , nrounds = 2L
    , verbose = 0L
)

print("Train lightgbm with verbose 1, print evaluation metric")
bst <- lightgbm(
    data = dtrain
    , params = train_params
    , nrounds = 2L
    , verbose = 1L
)

print("Train lightgbm with verbose 2, also print information about tree")
bst <- lightgbm(
    data = dtrain
    , params = train_params
    , nrounds = 2L
    , verbose = 2L
)

# You can also specify data as file path to a LibSVM/TCV/CSV format input
# Since we do not have this file with us, the following line is just for illustration
# bst <- lightgbm(
#     data = "agaricus.train.svm"
#     , num_leaves = 4L
#     , learning_rate = 1.0
#     , nrounds = 2L
#     , objective = "binary"
# )

#--------------------Basic prediction using lightgbm--------------
# You can do prediction using the following line
# You can put in Matrix, sparseMatrix, or lgb.Dataset
pred <- predict(bst, test$data)
err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error=", err))

#--------------------Save and load models-------------------------
# Save model to binary local file
lgb.save(bst, "lightgbm.model")

# Load binary model to R
bst2 <- lgb.load("lightgbm.model")
pred2 <- predict(bst2, test$data)

# pred2 should be identical to pred
print(paste("sum(abs(pred2-pred))=", sum(abs(pred2 - pred))))

#--------------------Advanced features ---------------------------
# To use advanced features, we need to put data in lgb.Dataset
dtrain <- lgb.Dataset(data = train$data, label = train$label, free_raw_data = FALSE)
dtest <- lgb.Dataset.create.valid(dtrain, data = test$data, label = test$label)

#--------------------Using validation set-------------------------
# valids is a list of lgb.Dataset, each of them is tagged with name
valids <- list(train = dtrain, test = dtest)

# To train with valids, use lgb.train, which contains more advanced features
# valids allows us to monitor the evaluation result on all data in the list
print("Train lightgbm using lgb.train with valids")
bst <- lgb.train(
    data = dtrain
    , params = train_params
    , nrounds = 2L
    , valids = valids
)

# We can change evaluation metrics, or use multiple evaluation metrics
print("Train lightgbm using lgb.train with valids, watch logloss and error")
bst <- lgb.train(
    data = dtrain
    , params = train_params
    , nrounds = 2L
    , valids = valids
    , eval = c("binary_error", "binary_logloss")
)

# lgb.Dataset can also be saved using lgb.Dataset.save
lgb.Dataset.save(dtrain, "dtrain.buffer")

# To load it in, simply call lgb.Dataset
dtrain2 <- lgb.Dataset("dtrain.buffer")
bst <- lgb.train(
    data = dtrain2
    , params = train_params
    , nrounds = 2L
    , valids = valids
)

# information can be extracted from lgb.Dataset using get_field()
label <- get_field(dtest, "label")
pred <- predict(bst, test$data)
err <- as.numeric(sum(as.integer(pred > 0.5) != label)) / length(label)
print(paste("test-error=", err))

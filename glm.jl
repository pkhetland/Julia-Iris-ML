# Import Dataset
using CSV, DataFrames

println("Importing dataset...")
iris = DataFrame(CSV.File("../../backup_datasets/iris/iris.csv"))
species = iris[:, :species]

# Preprocessing
using Lathe

scaled_feature = Lathe.preprocess.OneHotEncode(iris, :species)  # Perform OH encoding
iris = select!(iris, Not([:species]))  # Remove original species column
first(iris, 5)

# Train/test
using Random

sample = randsubseq(1:size(iris, 1), 0.75)
train = iris[sample, :]
notsample = [i for i in 1:size(iris, 1) if isempty(searchsorted(sample, i))]
test = iris[notsample, :]

y_test = species[notsample]

# Train GLM
using DataFrames, GLM

println("Training model...")

fm_setosa = @formula(setosa ~ sepal_length + sepal_width + petal_length + petal_width)
lm_setosa = glm(fm_setosa, train, Binomial(), LogitLink())
pred_setosa = predict(lm_setosa, test)

fm_virginica = @formula(virginica ~ sepal_length + sepal_width + petal_length + petal_width)
lm_virginica = glm(fm_virginica, train, Binomial(), LogitLink())
pred_virginica = predict(lm_virginica, test)

fm_versicolor = @formula(versicolor ~ sepal_length + sepal_width + petal_length + petal_width)
lm_versicolor = glm(fm_versicolor, train, Binomial(), LogitLink())
pred_versicolor = predict(lm_versicolor, test)

preds = hcat(pred_setosa, pred_virginica, pred_versicolor)

# Reclass

# Reclass by maximum predicted proba
# Reclass by maximum predicted probability
preds_cat = String[];
for i in 1:nrow(DataFrame(preds))
    if pred_setosa[i] >= pred_virginica[i] && pred_setosa[i] >= pred_versicolor[i]
        preds_cat = vcat(preds_cat ,"setosa")
    elseif pred_versicolor[i] >= pred_virginica[i] && pred_versicolor[i] >= pred_setosa[i]
        preds_cat = vcat(preds_cat ,"versicolor")
    elseif pred_virginica[i] >= pred_versicolor[i] && pred_virginica[i] >= pred_setosa[i]
        preds_cat = vcat(preds_cat ,"virginica")
    end
end
# Compute accuracy of GLM

correct = 0

n = length(y_test)
for i in 1:n
    if y_test[i] == preds_cat[i]
        correct += 1
    end
end

println(correct / n)

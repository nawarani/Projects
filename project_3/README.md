### Project 3

Create a **small** Flask App that captures user input and generates a prediction.



**TASK**

1. Choose from the following three datasets:

- A - [World Happiness](https://www.kaggle.com/PromptCloudHQ/world-happiness-report-2019) ~ Predict Ladder
- B - [Japanese Hostels](https://www.kaggle.com/koki25ando/hostel-world-dataset) ~ Predict Price
- C - [Chocolate](https://www.kaggle.com/rtatman/chocolate-bar-ratings) ~ Predict Review Score

2. Build a model on top of this data.
3. Save the model.
4. Wrap your saved model in a small Flask wrapper.
5. Have users input different X values to generate new predictions.



**RUBRIC**

Your project (model and Flask App) must:

- establish a benchmark and a naive model

- use a `LinearRegression`, 1 of `Lasso`/`Ridge`/`ElasticNet`, and 1 CatBoostRegressor/XGBoostRegressor
- have evidence of gridsearching
- use `sklearn.pipeline`
- accept user input and be able to generate new predictions on the fly
- use a 3rd-party python library/package that we haven't discussed.



**OPTIONAL**

If you've crushed the required bits with time to spare think about:

- adding a more *pretty* interface to your model/flask app using HTML/CSS
- hosting the app on heroku (or something similar)

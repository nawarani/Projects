### Project 3

Created a Flask App that captures user input and generates a prediction.



**TASK**

1. Data: [Chocolate](https://www.kaggle.com/rtatman/chocolate-bar-ratings) ~ Predicting Review Score
2. Built a model on top of this data.
4. Wrapped the saved model in a small Flask wrapper.
5. Set up user inputs for different X values to generate new predictions.


Details:
- established a benchmark and a naive model
- used a `LinearRegression`, `Lasso` and CatBoostRegressor
- gridsearched
- used `sklearn.pipeline`
- adding an html/css based interface to your model/flask app
- hosted the app on [heroku](https://chocolaterating.herokuapp.com/)

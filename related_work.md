# Related Work

### SVM sentiment analysis
After collecting all the data we need, we proposed to use students `comment` to predict how popular a professor is. Here are several steps we would do:
- we would need to split the entire dataset ***(clean_ratings.csv)*** into training set and testing set. 
- we would manually label the training set data identifying how poplular each professor is by using `avgRating` parameter in ***(clean_prof_info.csv)*** :
- Since this parameter is numerical, we would set a threshhold to indentify (popular, normal, not poplular). _Note: TBD_
- Vectorize the original comments by calculating **TF-IDF** and convert comments into a weighting matrix.
- Creating a Linear SVM Model, referencing [Sentiment Analysis using SVM](https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1)
- Lastly, we would test how good our model is by predicting the rest of testing data and compare the result with the original `avgRating` parameter and threshold.

### Other related reference:
https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1
https://aclanthology.org/W04-3253.pdf
https://link.springer.com/article/10.1007/s42452-020-2266-6
https://github.com/Nobelz/RateMyProfessorAPI (_Note: we didn't use any code from this repo, but this is still a good reference_)



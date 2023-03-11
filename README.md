# EECS 486 RMP Webscrape

## Acknowledgement

[RateMyProfessorAPI](https://github.com/Nobelz/RateMyProfessorAPI) authored by NobelZ, ChrisBryann, Ozeitis.

## Steps

- Get all Michigan prof IDs.
- [RMPAPI](https://github.com/Nobelz/RateMyProfessorAPI) provides ready to go GraphQL queries to get summary statistics and all ratings/comments. We will need to rewrite/customize most of the code.
- Use Pandas to clean and convert to CSVs.

## RMPAPI Details

- See comments in files for parts that can be reused.
- See `ratings_info.json` and `ratings.json` for sample JSON responses.
- No `pyproject.toml` or similar so cannot import locally.
- The queries can be ported. Rewrite to store data as dicts / JSON rather than class objects for easier Pandas convertion. ([`pd.Dataframe()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) can read a list of dicts or eq. JSON directly)

## Todo

- Investigate getting all professor IDs under the same school ID (Michigan: 1258)

  Possible Approaches:

  - Go to Michigan's school page, click on All Professors, use a Selenium script to load all avaiable professors, then scrape

  - GraphQL approach (emailed API repo author)

  - Manually click the button ~500 times (unironically might be the fastest way)

  - Keep making requests with https://www.ratemyprofessors.com/school?sid=[some_number] and keep incrementing [some_number] from 1 or something. Returns 8 professors ranked by teacher id (assigned upon professor profile creation), and then request more info about that professor with https://www.ratemyprofessors.com/professor?tid=[teacher_id]

  - ?

## Proposed CSV schema based on JSON

### professors.csv

| **Column Name**   | **Data Type** | **Note**                            |
| ----------------- | ------------- | ----------------------------------- |
| profID            | int           | Not in JSON                         |
| firstName         | str           |                                     |
| lastName          | str           |                                     |
| fullName          | str           | Parse from JSON first and last name |
| department        | str           |                                     |
| numRatings        | int           |                                     |
| wouldTakeAgainPct | float         |                                     |
| avgDifficulty     | float         |                                     |
| avgRating         | float         |                                     |

### ratings.csv

| **Column Name**     | **Data Type** | **Note**                                             |
| ------------------- | ------------- | ---------------------------------------------------- |
| profID              | int           | Not in JSON                                          |
| class               | str           |                                                      |
| attendanceMandatory | bool          | {"non mandatory", "mandatory"}, convert to bool      |
| comment             | str           | NAs dropped                                          |
| date                | `pd.datetime` | Convert from string (UTC format)                     |
| difficutyRating     | float         |                                                      |
| grade               | str           |                                                      |
| helpfulRating       | float         |                                                      |
| isForCredit         | bool          |                                                      |
| isForOnlineClass    | bool          |                                                      |
| ratingTags          | list          | Given in JSON as "tag1--tag2--tag3", convert to list |
| wouldTakeAgain      | bool          | Convert from int {0, 1}                              |

## Related Work

- For quick overview: https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1
- [Sentiment analysis using support vector machines with diverse information
  sources](https://aclanthology.org/W04-3253.pdf)
- To justify which type of SVM? [Multi-category news classification using Support Vector Machine based classifiers](https://link.springer.com/article/10.1007/s42452-020-2266-6)

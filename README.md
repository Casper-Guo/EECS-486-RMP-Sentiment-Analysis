# EECS 486 RMP Webscrape

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

  - ?

## Proposed CSV schema based on JSON

### professors.csv

- profID (int) (not in JSON)
- firstName (str)
- lastName (str)
- fullName (str) (not in JSON)
- numRatings (int)
- wouldTakeAgainPct (int)
- avgDifficulty (float)
- avgRating (float)

### ratings.csv

- profID (int) (not in JSON)
- class (str)
- comment (str)
- attendanceMandatory (bool)
- date (`pd.datetime`)
- difficultyRating (float) (Older RMP allows .5 ratings)
- grade (str)
- helpfulRating (float)
- isForCredit (bool)
- ifForOnlineClass (bool)
- ratingTags (str) (extra processing to convert to list)
- wouldTakeAgain (bool) (represented in JSON as int)

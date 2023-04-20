# EECS 486 RMP Webscrape

Using all reviews given to University of Michigan professors on [ratemyprofessors.com](ratemyprofessors.com) (RMP) as the corpus, we benchmark the performance of different text-encoding methodologies and machine learning models.

## Installation
You should have a Jupyter development set up, preferably using Python 3.11 kernel.

Then, `pip install -r requirements.txt`

## Directory Structure
```
.

├── Data
│   ├── clean_prof_info.csv
│   ├── clean_ratings.csv
│   ├── data_cleaning.ipynb
│   ├── glove.6B.100d.txt
│   ├── glove.6B.100d.txt.word2vec
│   ├── raw_prof_info.csv
│   └── raw_ratings.csv
├── README.md
├── RateMyProfessorAPI
│   ├── LICENSE
│   ├── MANIFEST.in
│   ├── README.md
│   ├── examples
│   │   └── example.py
│   ├── ratemyprofessor
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── professor.cpython-38.pyc
│   │   │   └── school.cpython-38.pyc
│   │   ├── json
│   │   │   ├── header.json
│   │   │   ├── professorquery.json
│   │   │   └── ratingsquery.json
│   │   ├── professor.py
│   │   ├── ratings.json
│   │   ├── ratings_info.json
│   │   ├── sample.py
│   │   └── school.py
│   ├── requirements.txt
│   ├── setup.cfg
│   ├── setup.py
│   └── tests
│       └── test.py
├── data_acquisition.py
├── experiment.ipynb
├── html
│   ├── diff.txt
│   ├── html.txt
│   └── profID.txt
├── pipeline.ipynb
├── requirements.txt
├── scraper.py
└── util.py
```

## File Overview
### `Data/`
- Raw and cleaned datasets, dataset schema is provided below.
- `data_cleaning.ipynb`: Processing procedures to clean raw data.
- `glove.6B.100d.txt`: Download available at the [GloVe website](https://nlp.stanford.edu/projects/glove/). We used the 6B tokens dataset with 100-dimension vectors.

### `RateMyProfessorAPI`/
Lightly edited fork of [RateMyProfessorAPI](https://github.com/Nobelz/RateMyProfessorAPI), see [acknowledgement](#Acknowledgement).

### `html`/
Webscrapping artifacts.

### `./`
- `data_acquisiton.py`: Given a list of profIDs, use RMPAPI to retrieve relevant information in JSON format.
- `experiment.ipynb`: Benchmarking with different encodings and machine learning models.
- `scraper.py`: Selenium program to retrieve UMich profIDs.
- `util.py`: Project utilities.

## Cleaned Dataset Schema

### `clean_prof_info.csv`

| **Column Name**   | **Data Type** | **Note**                            |
| ----------------- | ------------- | ----------------------------------- |
| profID            | int           |                                     |
| firstName         | str           |                                     |
| lastName          | str           |                                     |
| fullName          | str           | Concatenate first and last name     |
| department        | str           | Known defects, not reliable         |
| numRatings        | int           |                                     |
| wouldTakeAgainPct | float         | Ranges from 0 to 100, has Na        |
| avgDifficulty     | float         | Ranges from 1 to 5                  |
| avgRating         | float         | Ranges from 1 to 5                  |

### `clean_ratings.csv`

| **Column Name**     | **Data Type** | **Note**                                             |
| ------------------- | ------------- | ---------------------------------------------------- |
| profID              | int           |                                                      |
| class               | str           |                                                      |
| attendanceMandatory | bool          |                                                      |
| comment             | str           |                                                      |
| date                | `pd.datetime` | UTC format, accurate to second                       |
| difficutyRating     | float         | Range from 1 to 5                                    |
| grade               | str           | Letter grades, with +/-                              |
| helpfulRating       | float         | Range from 1 to 5                                    |
| isForCredit         | bool          |                                                      |
| isForOnlineClass    | bool          |                                                      |
| ratingTags          | list          | List of up to 3 tags                                 |
| wouldTakeAgain      | bool          |                                                      |

Earlier comments may have `difficultyRating` and `helpfulRating` at 0.5 increments.

At some point, the site began to only allow integer ratings.

## Acknowledgement

[RateMyProfessorAPI](https://github.com/Nobelz/RateMyProfessorAPI) authored by NobelZ, ChrisBryann, Ozeitis. Apache-2.0 license.
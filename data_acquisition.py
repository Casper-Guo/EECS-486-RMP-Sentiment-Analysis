"""
Expected input/commandline argument:
A file containing a list of teacher IDs, one per line.

Use GraphQL query provided in RMPAPI to get two JSON files for each prof.
One containing a list of ratings, and the other containing information about the prof.
(See sample JSONs in RateMyPrefessorAPI/ratemyprofessor)

Output:
Two CSV files corresponding to the two types of JSON responses.
No cleaning and processing is performed.
"""

import requests
import json
import base64
import sys
import time
import pandas as pd
from pathlib import Path


with open(Path("RateMyProfessorAPI/ratemyprofessor/json/ratingsquery.json"), 'r') as f:
    ratings_query = json.load(f)

with open(Path("RateMyProfessorAPI/ratemyprofessor/json/professorquery.json"), 'r') as f:
    prof_query = json.load(f)

with open(Path("RateMyProfessorAPI/ratemyprofessor/json/header.json"), 'r') as f:
    headers = json.load(f)


def get_prof_info(profID: int) -> dict:
    """
    Returns: dict
    {
    profID: int,
    avgDifficulty: float,
    avgRating: float,
    department: str,
    firstName: str,
    lastName: str
    numRatings: int,
    wouldTakeAgainPercent: int
    }
    """
    headers["Referer"] = f"https://www.ratemyprofessors.com/ShowRatings.jsp?tid={profID}"

    prof_query["variables"]["id"] = base64.b64encode(("Teacher-%s" % profID)
                                                     .encode('ascii')).decode('ascii')
    data = requests.post(
        url="https://www.ratemyprofessors.com/graphql", json=prof_query, headers=headers)

    if data is None or json.loads(data.text)["data"]["node"] is None:
        raise ValueError(
            "Professor not found with that id or bad request.")

    # Process the data JSON and return that directly
    prof_data = json.loads(data.text)["data"]["node"]
    response = {}

    response["profID"] = profID
    response["avgDifficulty"] = prof_data.get("avgDifficulty")
    response["avgRating"] = prof_data.get("avgRating")
    response["department"] = prof_data.get("department")
    response["firstName"] = prof_data.get("firstName")
    response["lastName"] = prof_data.get("lastName")
    response["numRatings"] = prof_data.get("numRatings")
    response["wouldTakeAgainPercent"] = prof_data.get("wouldTakeAgainPercent")

    return response


def get_ratings(profID: int, num_ratings: int) -> list:
    """
    Returns: list of dicts
    {
    profID: int,
    attendanceMandatory: str,
    class: str,
    comment: str,
    date: str,
    difficultyRating: int,
    grade: str,
    helpfulRating: int,
    isForCredit: str,
    isForOnlineClass: str,
    ratingTags: str,
    wouldTakeAgainPercent: bool
    }
    """
    if num_ratings == 0:
        return []

    headers["Referer"] = f"https://www.ratemyprofessors.com/ShowRatings.jsp?tid={profID}"

    ratings_query["variables"]["id"] = base64.b64encode(
        (f"Teacher-{profID}").encode('ascii')).decode('ascii')

    ratings_query["variables"]["count"] = num_ratings

    data = requests.post(
        url="https://www.ratemyprofessors.com/graphql", json=ratings_query, headers=headers)

    if data is None or json.loads(data.text)["data"]["node"]["ratings"]["edges"] is None:
        return []

    # Similar, process the JSON and return that directly
    ratings_data = json.loads(
        data.text)["data"]["node"]["ratings"]["edges"]
    ratings = []

    for rating_data in ratings_data:
        data = rating_data["node"]
        rating = {}

        rating["profID"] = profID
        rating["attendanceMandatory"] = data.get("attendanceMandatory")
        rating["class"] = data.get("class")
        rating["comment"] = data.get("comment")
        rating["date"] = data.get("date")
        rating["difficultyRating"] = data.get("difficultyRating")
        rating["grade"] = data.get("grade")
        rating["helpfulRating"] = data.get("helpfulRating")
        rating["isForCredit"] = data.get("isForCredit")
        rating["isForOnlineClass"] = data.get("isForOnlineClass")
        rating["ratingTags"] = data.get("ratingTags")

        if data["wouldTakeAgain"] == 1:
            rating["wouldTakeAgain"] = True
        elif data["wouldTakeAgain"] == 0:
            rating["wouldTakeAgain"] = False
        else:
            rating["wouldTakeAgain"] = None

        ratings.append(rating)

    return ratings


def main():
    assert len(
        sys.argv) == 2, "Supply a text file containing a list of teacher IDs"

    ID_file = Path(sys.argv[1])
    profs_info = []
    ratings = []

    with open(ID_file, "r") as f:
        IDs = f.readlines()
        IDs = [int(id.strip()) for id in IDs]

    for count, ID in enumerate(IDs):
        try:
            prof_info = get_prof_info(ID)

            num_ratings = prof_info.get("numRatings")
            print(count, prof_info.get("lastName"))

            prof_ratings = get_ratings(ID, num_ratings)
            profs_info.append(prof_info)
            ratings.extend(prof_ratings)
        except:
            continue

    df_info = pd.DataFrame(profs_info)
    df_ratings = pd.DataFrame(ratings)

    df_info.to_csv(Path("Data/raw_prof_info.csv"), index=False)
    df_ratings.to_csv(Path("Data/raw_ratings.csv"), index=False)

    return 0


if __name__ == "__main__":
    main()

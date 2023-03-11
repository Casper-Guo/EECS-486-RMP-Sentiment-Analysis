"""
Various project utilities.
"""

import pandas as pd
from bs4 import BeautifulSoup as bs


def get_profID(html: str) -> list:
    """
    Args:
        html (str): the html of the 'all professors' page

    Returns:
        A list of contained profIDs.

        This is expected in the href attributes of the TeacherCard classes.
    """

    soup = bs(html, "html.parser")

    teacher_cards = soup.find_all(
        "a", {"class": lambda x: x and x.startswith("TeacherCard__StyledTeacherCard")})
    links = [card.attrs["href"] for card in teacher_cards]

    IDs = [link[link.find("?tid=")+5:] for link in links]

    return IDs


def find_diff(received_IDs: list, all_IDs: list) -> list:
    return set(all_IDs) - set(received_IDs)


def main():
    with open("profID.txt", "r") as f:
        all_IDs = f.readlines()
        all_IDs = [int(id.strip()) for id in all_IDs]

    df_prof = pd.read_csv("Data/raw_prof_info.csv")
    received_IDs = df_prof["profID"].to_list()

    diff = find_diff(received_IDs, all_IDs)

    with open("diff.txt", "w") as f:
        for id in diff:
            f.write(str(id) + "\n")

    return 0


if __name__ == '__main__':
    main()

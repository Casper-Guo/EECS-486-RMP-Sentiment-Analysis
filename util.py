"""
Various project utilities.
"""

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


def main():
    with open("html.txt", 'r') as f:
        html = f.read()

    # with open("profID.txt", 'w+') as f:
    #     for profID in get_profID(html):
    #         f.write(profID + "\n")

    return 0


if __name__ == '__main__':
    main()

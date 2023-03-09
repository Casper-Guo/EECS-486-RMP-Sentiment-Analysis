from professor import Professor
import json

juett = Professor(2048250)
ratings_info = juett._get_rating_info(2048250)

with open("ratings_info.json", "w") as f:
    f.write(json.dumps(ratings_info, indent=4))

ratings = juett.get_ratings()

with open("ratings.json", "w") as f:
    f.write(json.dumps(ratings, indent=4))

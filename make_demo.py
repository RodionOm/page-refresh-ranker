"""Write app/static/demo_pages.csv: synthetic pages in the website's upload format (safe to publish)."""
from ranker.features import to_prediction_format
from ranker.synthetic import make_export

if __name__ == "__main__":
    to_prediction_format(make_export(400, seed=3)).to_csv("app/static/demo_pages.csv", index=False)
    print("wrote app/static/demo_pages.csv")

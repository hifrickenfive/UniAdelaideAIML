from pathlib import Path

path = Path(__file__).parent / "../data/test.csv"
with path.open() as f:
    test = list(csv.reader(f))
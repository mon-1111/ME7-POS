import csv
import os

# Path to items.csv relative to this Python file (works on Mac, Jetson, Docker)
CSV_PATH = os.path.join(os.path.dirname(__file__), "items.csv")


class ItemCatalog:
    def __init__(self, csv_path: str = CSV_PATH):
        self.items = {}
        self._load(csv_path)

    def _load(self, csv_path: str):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Handle weird BOM or header formatting
            fieldnames = reader.fieldnames

            # Identify columns safely
            id_col = next(fn for fn in fieldnames if "Class ID" in fn)
            name_col = next(fn for fn in fieldnames if "Class Name" in fn)
            product_col = next(fn for fn in fieldnames if "Product" in fn)
            price_col = next(fn for fn in fieldnames if "Price" in fn)

            for row in reader:
                cid = int(row[id_col])
                self.items[cid] = {
                    "class_name": row[name_col],
                    "product": row[product_col],
                    "price": float(row[price_col]),
                }

    def get(self, class_id: int):
        """Return dict with class_name, product, price or None."""
        return self.items.get(class_id)


# Manual test
if __name__ == "__main__":
    catalog = ItemCatalog()
    print(f"Loaded {len(catalog.items)} items\n")

    for cid in range(5):
        print(cid, "->", catalog.get(cid))
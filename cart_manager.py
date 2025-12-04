from collections import defaultdict
from csv_manager import ItemCatalog


class CartManager:
    def __init__(self, catalog: ItemCatalog):
        self.catalog = catalog
        self.items = defaultdict(int)  # class_id -> quantity

    def clear(self):
        """Empty the cart."""
        self.items.clear()

    def add_item(self, class_id: int, qty: int = 1):
        """Add qty of a class_id to the cart."""
        self.items[class_id] += qty

    def get_lines(self):
        """
        Return a list of lines for the receipt.
        Each line: {class_id, product, qty, price, subtotal}
        """
        lines = []
        for cid, qty in self.items.items():
            meta = self.catalog.get(cid)
            if not meta:
                continue
            price = meta["price"]
            subtotal = price * qty
            lines.append({
                "class_id": cid,
                "product": meta["product"],
                "qty": qty,
                "price": price,
                "subtotal": subtotal,
            })
        return lines

    def get_total(self):
        return sum(line["subtotal"] for line in self.get_lines())


# Quick manual test
if __name__ == "__main__":
    catalog = ItemCatalog()
    cart = CartManager(catalog)

    # Simulate scanning some items
    cart.add_item(0)   # coffee_nescafe
    cart.add_item(3)   # coke-in-can
    cart.add_item(3)   # another coke-in-can
    cart.add_item(31)  # meadows_truffle_chips

    print("RECEIPT:")
    for line in cart.get_lines():
        print(f"{line['product']:30} x{line['qty']:2}  "
              f"@ {line['price']:5.2f}  = {line['subtotal']:6.2f}")

    print(f"\nTOTAL: {cart.get_total():.2f}")
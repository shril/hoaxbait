from predict_flask import classify

body = "Al-Sisi has denied Israeli reports stating that he offered to extend the Gaza Strip."
head = "Apple installing safes in-store to protect gold Watch Edition"

print (classify([head, body]))
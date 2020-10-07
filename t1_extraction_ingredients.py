# -*- coding: utf-8 -*-
import re

ingredients_fn = "./data/ingredients.txt"

pattern = "^((\d*((,\d+)|( à \d+)?(\/\d+)|)( à \d*((,\d+)|( à \d+)?(\/\d+)|))?|½)? *((ml|mL|cl|gallon|lb|g|kg|oz|(Q|q)uelques|(P|p)incées?|(E|e)nveloppes?|(Z|z)estes?|(T|t)raits?|(M|m)orceaux?|(P|p)intes?|(T|t)asses?|(B|b)oîtes?|(F|f)euilles?|(R|r)ondelles?|(G|g)ousses?|(T|t)ranches?|(L|l)amelles?|((cuillères?|c\.|\.c|\.) ?à ?(café|thé|soupe|\.s|s\.|\.c|c\.))) )? *(\([^)]+\))?) *(?:de|d')?(.*)$";
regex_quantity = re.compile(pattern)


def load_ingredients(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_items = f.readlines()
    ingredients = [x.strip() for x in raw_items]
    return ingredients


def get_ingredients(text):
    return regex_quantity.match(text).group(1).strip(), regex_quantity.match(text).group(32).strip()


if __name__ == '__main__':
    print("Lecture des ingrédients du fichier {}.".format(ingredients_fn))
    all_items = load_ingredients(ingredients_fn)

    for item in all_items:
        quantity, ingredient = get_ingredients(item)
        if item != "":
            print("{}   QUANTITE:{}   INGREDIENT:{}".format(item, quantity, ingredient))

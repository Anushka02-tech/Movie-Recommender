import pandas as pd
import numpy as np
import itertools

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery",
    "Quinn", "Reese", "Rowan", "Skyler", "Dakota", "Emerson", "Finley", "Harper",
    "Kai", "Logan", "Parker", "Sage", "Blake", "Cameron", "Drew", "Elliot",
    "Hayden", "Jesse", "Kendall", "Micah", "Noel", "Peyton", "Remy", "Shawn",
    "Adrian", "Bailey", "Charlie", "Devon", "Eden", "Frankie", "Gray", "Indigo",
    "Amara", "Beau", "Celeste", "Darius", "Estelle", "Felix", "Giana", "Hugo",
    "Ines", "Jasper", "Kiran", "Lena", "Marcus", "Nadia", "Oscar", "Priya",
    "Quincy", "Ravi", "Selena", "Theo", "Uma", "Viggo", "Willa", "Xander",
    "Yara", "Zane", "Aria", "Bodhi", "Clara", "Dante", "Esme", "Felipe",
    "Greta", "Hana", "Ivan", "Jolene", "Kian", "Luna", "Mateo", "Nina",
    "Omar", "Petra", "Quill", "Rosa", "Sami", "Tara", "Umar", "Vera",
    "Wren", "Ximena", "Yusuf", "Zola", "Anwen", "Bram", "Coral", "Denny",
    "Elio", "Fiora", "Gideon", "Halle", "Ilan", "Junia", "Koa", "Lior",
    "Maren", "Niko", "Orla", "Pia", "Qadir", "Roux", "Soren", "Talia",
]

LAST_NAMES = [
    "Rivera", "Chen", "Patel", "Kowalski", "Nguyen", "Okafor", "Martinez", "Kim",
    "Johansson", "Silva", "Andersson", "Haddad", "Petrov", "Yamamoto", "Costa",
    "Novak", "Singh", "Dubois", "Fernandez", "Larsen", "Moreau", "Osei",
    "Rossi", "Sato", "Tran", "Weber", "Zhang", "Alvarez", "Berg", "Castillo",
    "Adebayo", "Bianchi", "Carvalho", "Dahl", "Ekwueme", "Farhat", "Giannis", "Hoang",
    "Ibarra", "Jansen", "Kaur", "Lindqvist", "Mensah", "Nakamura", "Ortiz", "Papadopoulos",
    "Quiroga", "Ramirez", "Suzuki", "Tanaka", "Ueda", "Valdez", "Wojcik", "Xiong",
    "Yilmaz", "Zeleny", "Abara", "Barros", "Cohen", "Delgado", "Esposito", "Farah",
    "Gorski", "Herrera", "Ilic", "Jovanovic", "Kallas", "Lund", "Mbeki", "Nowak",
    "Ostrowski", "Pham", "Qureshi", "Reyes", "Santos", "Toivonen", "Umeh", "Vukovic",
    "Wilder", "Ximenes", "Yildiz", "Zamora", "Aoki", "Baptiste", "Correa", "Duarte",
    "Ekstrom", "Falk", "Guo", "Hernandez", "Iyer", "Jaramillo", "Kobayashi", "Leung",
]

user_ids = pd.read_csv('user_ids.csv')['user_id'].values
print(f"Generating display names for {len(user_ids)} users...")

all_combos = list(itertools.product(FIRST_NAMES, LAST_NAMES))
pool_size = len(all_combos)
print(f"Name pool size: {pool_size:,} combinations for {len(user_ids):,} users "
      f"({pool_size / len(user_ids):.1f}x headroom)")

if pool_size < len(user_ids):
    raise ValueError(
        f"Name pool ({pool_size}) is smaller than number of users ({len(user_ids)}). "
        f"Add more names to FIRST_NAMES/LAST_NAMES."
    )

rng = np.random.RandomState(seed=42)  # fixed seed -> reproducible, but shuffled-looking
shuffled_indices = rng.permutation(pool_size)[:len(user_ids)]
names = [f"{all_combos[i][0]} {all_combos[i][1]}" for i in shuffled_indices]

user_names_df = pd.DataFrame({'user_id': user_ids, 'display_name': names})
user_names_df.to_csv('user_display_names.csv', index=False)

print(f"Saved user_display_names.csv")
print(f"\nSample:")
print(user_names_df.head(10))
print(f"\nUnique names generated: {user_names_df['display_name'].nunique()} / {len(user_names_df)}")
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT = Path("data/house_price_generated.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
n=1000
#area_sqft,bedrooms,bathrooms,house_age,price
area_sqft = rng.integers(1000,2500,n)
bedrooms =  rng.integers(2,4,n)
bathrooms  =  rng.integers(1,3,n)
house_age =  np.round(rng.uniform(2,10,n), 2)
#price = rng.uniform(3500000,9500000,n)


noise = rng.uniform(0,5,n)

price = (
    3500000.0                   # Base price 3500000 
    + 0.15 * area_sqft          
    + 1500000.0 * bedrooms           # Adds 1500000 per bedroom
    + 100000.0 * bathrooms          # Adds 100000k per bathroom
    - 200000.0 * house_age           # Subtracts 2k per year of age
)

price = np.clip(price, 3500000, 10000000)  


df = pd.DataFrame({
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "house_age": house_age,
    "price": price,
})

df.to_csv(OUTPUT, index=False)

print(f"Saved {len(df)} rows to {OUTPUT}")
print(df.head())
print(df.describe())
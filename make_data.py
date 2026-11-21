from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
df.to_csv("data/housing.csv", index=False)
print("Saved data/housing.csv", df.shape)

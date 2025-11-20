from model4.package import load_saved_model

data = load_saved_model(
    saved_dir='model4/saved_model'
)

print(data)

vectorizer = data['embedding']
model = data['model']

sample = vectorizer.transform(["I love this product!", "Terrible service."])

predictions = model.predict(sample)
print(predictions)

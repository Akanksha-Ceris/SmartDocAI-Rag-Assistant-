import google.generativeai as genai
genai.configure(api_key="AIzaSyA-ccOdWkpq-1zGbrV1cQp1e1Ehbfp1M5c")
# Here is the list of all available models
models = genai.list_models()
for model in models:
    print(model.name)

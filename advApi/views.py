import os
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from advMlApi.settings import BASE_DIR
import joblib
import json

# Load the model once at startup
model = joblib.load(os.path.join(BASE_DIR, 'advApi', 'rfc_model.pkl'))

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Single row prediction
            if 'features' in data:
                features = data['features']
                if not isinstance(features, list) or len(features) != 209 or not all(isinstance(x, (int, float)) for x in features):
                    return JsonResponse({"error": "Expected exactly 209 numeric feature values."}, status=400)
                prediction = model.predict([features])
                result = "Ad" if prediction[0] == 1 else "Non-Ad"
                return JsonResponse({"prediction": result})

            # Multiple rows prediction
            elif 'features_list' in data:
                features_list = data['features_list']
                if not isinstance(features_list, list):
                    return JsonResponse({"error": "'features_list' must be a list of 209-length feature lists."}, status=400)

                for row in features_list:
                    if not isinstance(row, list) or len(row) != 209 or not all(isinstance(x, (int, float)) for x in row):
                        return JsonResponse({"error": "All rows must contain exactly 209 numeric values."}, status=400)

                predictions = model.predict(features_list)
                results = ["Ad" if p == 1 else "Non-Ad" for p in predictions]
                return JsonResponse({"predictions": results})

            return JsonResponse({"error": "Invalid input. Provide either 'features' or 'features_list'."}, status=400)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return render(request, "adPred.html")

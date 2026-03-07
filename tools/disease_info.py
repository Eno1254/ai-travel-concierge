def disease_info(query):

    data = {
        "dengue": "Dengue is a mosquito-borne viral infection causing fever, headache, and joint pain.",
        "malaria": "Malaria is a disease caused by parasites transmitted through mosquito bites.",
        "diabetes": "Diabetes is a chronic disease that affects how the body processes blood sugar."
    }

    for disease in data:
        if disease in query.lower():
            return data[disease]

    return "No disease information found. Consult a doctor."
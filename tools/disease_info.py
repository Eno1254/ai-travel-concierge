def disease_info(disease):

    info = {
        "dengue": "Dengue is a mosquito-borne viral infection causing fever and joint pain.",
        "malaria": "Malaria is caused by parasites transmitted through mosquito bites.",
        "diabetes": "Diabetes is a chronic disease that affects how the body processes blood sugar."
    }

    return info.get(disease.lower(), "Consult a doctor for more information.")
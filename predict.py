import joblib
import pandas as pd
from typing import Dict
import ast

MODEL = joblib.load('stage_4/rfc_model.pkl')
MLB = joblib.load('stage_4/mlb.pkl')

def predict_profit(movie_data: Dict):
    df = pd.DataFrame([movie_data])
    tag_matrix = MLB.transform([movie_data['genres']])
    tag_df = pd.DataFrame(tag_matrix, columns=MLB.classes_, index=df.index)
    df = pd.concat([df, tag_df], axis=1)
    columns = df.select_dtypes(include='number').columns.tolist()
    columns_to_drop = ['revenue', 'id', 'made_profit', 'profit']
    columns = [col for col in columns if col not in columns_to_drop]
    data = df[columns]
    prediction = MODEL.predict(data)
    probability = MODEL.predict_proba(data)[:, prediction]
    return prediction[0], probability[0][0]


def main():
    data = input("Enter movie data in dictionary form: \n")
    data = ast.literal_eval(data)
    if type(data) != dict:
        raise TypeError("Invalid data type")
    prediction, probability = predict_profit(data)
    print("Movie will ", end='')
    if prediction == 1:
        print("make profit. ", end='')
    else:
        print("not make profit. ", end='')
    print(f"Confidence: {probability*100:.2f}%")

if __name__ == '__main__':
    main()
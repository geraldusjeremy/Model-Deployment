import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import OneHotEncoder

#inisiasi memanggil file pickle model dan nama-nama kolom, lalu membuat encoder
class Predictor:
    def __init__(self, model_path, training_columns_path):
        self.model = joblib.load(model_path)
        with open(training_columns_path, 'rb') as f:
            self.training_columns = pickle.load(f)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    #function untuk menyiapkan data agar struktur dan isi kolomnya sesuai dengan model saat dilatih
    def prepare_input(self, user_input_df, columns_to_encode):
        encoded_parts = []
        #looping untuk semua kolom kategorikal dilakukan encode pada kolom-kolomnya dan difit 
        for column in columns_to_encode:
            self.encoder.fit(user_input_df[[column]])
            encoded = self.encoder.transform(user_input_df[[column]])
            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{column}_{cat}" for cat in self.encoder.categories_[0]],
                index=user_input_df.index
            )
            user_input_df = pd.concat([user_input_df.drop(columns=[column]), encoded_df], axis=1)

        #samakan urutan & isi kolom dengan saat training
        user_input_df = user_input_df.reindex(columns=self.training_columns, fill_value=0)
        return user_input_df
    #function predict untuk memanggil model dan mengembalikan hasil canceled atau not canceled
    def predict(self, prepared_input_df):
        prediction = self.model.predict(prepared_input_df)
        return prediction

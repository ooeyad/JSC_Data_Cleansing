
from flask import Flask, jsonify
from flask_restful import Resource, Api
import numpy as np
import math
import csv
import pandas as pd
import os
import json
import datetime
from tqdm import tqdm
from pandas_profiling import ProfileReport
import webbrowser
import matplotlib.pyplot as plt

from recognizer.id_reader import read_name_img, read_id_img, read_dob_img, read_nationality_img
import glob
from pdf2image import convert_from_path
from queue import PriorityQueue as pq





app = Flask(__name__)
api = Api(app)

class Golden_Record(Resource):

    df = pd.DataFrame()
    with open("./data/countries.json", "r") as f:
        countries = json.load(f)
    print("countries length = " + str(len(countries)))


    def check_year_ID_list(self,id_tuple):
        dob = id_tuple[0]
        p_id = id_tuple[1]

        if dob == "":
            return ""

        #     try:
        day, month, year = dob.split("/")
        isValidDate = True

        try:
            datetime.datetime(int(year), int(month), int(day))
        except ValueError:
            isValidDate = False

        if isValidDate:
            year = year[-2:]
            ID = p_id
            ID_year = p_id[1:3]
            if ID_year == year:
                return dob
            else:
                return ""
        else:
            return ""

    def check_country_code_ID_list(self,id_nat_tuple):
        ID_country_code = id_nat_tuple[1][3:6]  # ID_NUMBER
        #     print(50*"*")
        #     print(row["NAT"],type(row["NAT"]))
        country = ""
        if id_nat_tuple[0] == "":
            for key in self.countries:
                if self.countries[key]['country-code'] == ID_country_code:
                    return str(key)
            # country = self.countries.get(str(id_nat_tuple[0]).lower())
            #     print(country,type(country))
            if country == "":
                return ""

            # country_code = country["country-code"]

        else:
            return id_nat_tuple[0].lower()

    def check_ID_pattern_list(self,x):
        return all(
            [
                (len(set(x)) > 1),
                len(x) == 11,
            ]
        )

    def score_tuple(self,rec_tuple):  # (FULL_NAME_ARA,MOBILE_NUMBER,NAT,Address,DOB)
        score = 0
        if rec_tuple[0] != "" and len(rec_tuple[0]) >= 3:
            score += 10
        if rec_tuple[1] != "" and len(rec_tuple[1]) >= 3:
            score += 2
        if rec_tuple[2] != "" and len(rec_tuple[2]) >= 3:
            score += 5
        if rec_tuple[3] != "" and len(rec_tuple[3]) >= 3:
            score += 2
        if rec_tuple[4] != "" and len(rec_tuple[4]) >= 3:
            score += 5

        return score


    def reformat_df(self,chosen_csv):
        df = pd.read_csv(chosen_csv, index_col=0)
        df = df[:20000]
        df["ID_NUMBER"] = df["ID_NUMBER"].fillna(0)
        df['DOB'] = df['DOB'].str.strip()
        df["DOB"] = df["DOB"].fillna('')
        df["NAT"] = df["NAT"].astype("str", errors="ignore")
        df['NAT'] = df['NAT'].str.strip()
        df["NAT"] = df["NAT"].fillna('')
        # df["NAT"] = df["NAT"].replace("nan", "")
        print("empty nats " + str(df['NAT'] == 'nan'))
        df['ID_NUMBER'] = df['ID_NUMBER'].astype(np.int64)
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str", errors="ignore")
        df['ID_NUMBER'] = df['ID_NUMBER'].str.strip()

        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df['MOBILE_NUMBER'] = df['MOBILE_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].fillna('')
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace("0.0", "").replace("nan", "")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace(".0", "").replace("nan", "")
        df["PDF_PATH"] = df["PDF_PATH"].astype("str", errors="ignore").replace("nan", "")
        df["ID_PATH"] = df["ID_PATH"].astype("str", errors="ignore").replace("nan", "")
        df = df.fillna("")

        df = df.reset_index(drop=True)

        return df

    def delete_col_df(self,chosen_csv):
        df = pd.read_csv(chosen_csv, index_col=0)
        df = df[:20000]
        df["ID_NUMBER"] = df["ID_NUMBER"].fillna(0)
        df['DOB'] = df['DOB'].str.strip()
        df["DOB"] = df["DOB"].fillna('')
        df["NAT"] = df["NAT"].astype("str", errors="ignore")
        df['NAT'] = df['NAT'].str.strip()
        df["NAT"] = df["NAT"].fillna('')
        # df["NAT"] = df["NAT"].replace("nan", "")
        print("empty nats " + str(df['NAT'] == 'nan'))
        df['ID_NUMBER'] = df['ID_NUMBER'].astype(np.int64)
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str", errors="ignore")
        df['ID_NUMBER'] = df['ID_NUMBER'].str.strip()

        df.drop(columns=["MOBILE_NUMBER], inplace=True)
        
        df["PDF_PATH"] = df["PDF_PATH"].astype("str", errors="ignore").replace("nan", "")
        df["ID_PATH"] = df["ID_PATH"].astype("str", errors="ignore").replace("nan", "")
        df = df.fillna("")

        df = df.reset_index(drop=True)

        return df
        
    def verify_cells(self):
        df = self.df

        print("Before check_id_pattern")
        res = df[(df['ID_NUMBER'].str.len() > 1) & (df['ID_NUMBER'].str.len() == 11)]
        # res = df.loc[df.apply(check_ID_pattern)]
        print("Before check_country_code")

        res['NAT'] = [self.check_country_code_ID_list(x) for x in zip(res['NAT'], res['ID_NUMBER'])]
        # res["NAT"] = res.apply(check_country_code_ID, axis=1)

        print("Before check_year")
        # res["DOB"] = res.apply(check_year_ID, axis=1)
        # res = df
        res["DOB"] = [self.check_year_ID_list(x) for x in zip(res["DOB"], res["ID_NUMBER"])]

        print("before applying score")
        # res["SCORE"] = res.apply(score_record, axis=1)
        res["SCORE"] = [self.score_tuple(x) for x in
                        zip(res["FULL_NAME_ARA"], res["MOBILE_NUMBER"], res["NAT"], res["Address"], res["DOB"])]

        print("after applying score")

        df.reset_index(inplace=True)
        print("after applying reset_index on df.")

        res.reset_index(inplace=True)

        # res = res.sort_values(['SCORE'],ascending=False)

        print("after applying reset_index on res.")

        # del df['index']
        # del res['index']
        # state.df = res
        # res.to_csv("./data/saved_df.csv",encoding='utf-8')
        self.df = res
        print("after saving df ")

    def fill_records(self):
        # df = pd.read_csv("./data/saved_df.csv")
        df = self.df
        sorted_res = df.groupby(["ID_NUMBER", "FULL_NAME_ARA"])\
            .apply(lambda x: x.sort_values(["ID_NUMBER"], ascending=False))\
            .reset_index(drop=True)

        # sorted_res = df.groupby(["ID_NUMBER", "FULL_NAME_ARA"]) \
        #     .apply(lambda x: x)\
        #     .reset_index(drop=True)

        # sorted_res = df.sort_values(by="SCORE", ascending=False).reset_index(drop=True)


        print("finished sorting res")
        # sorted_res = (
        #     state.df.groupby(["ID_NUMBER", "FULL_NAME_ARA"])
        #     .apply(lambda x: x.sort_values(["SCORE"], ascending=False))
        #     .reset_index(drop=True)
        # )
        grouped = sorted_res.groupby("ID_NUMBER")

        print("finished grouped")

        df_columns = sorted_res.columns
        filtered_df = pd.DataFrame(columns=df_columns)

        print("length of grouped : " + str(len(grouped)))

        for name, group in tqdm(grouped):

            if group.shape[0] > 1:
                base_row = group.iloc[0].copy()
                addresses = []
                numbers = []
                Date_of_birth = []
                nation = []
                for row_index, row in group.iterrows():
                    if row["Address"]:
                        addresses.append(row["Address"])
                    if row["MOBILE_NUMBER"]:
                        numbers.append(row["MOBILE_NUMBER"])
                    if row["DOB"]:
                        Date_of_birth.append(row["DOB"])
                    if row["NAT"]:
                        nation.append(row["NAT"])

                numbers = set(numbers)
                addresses = set(addresses)
                Date_of_birth = set(Date_of_birth)
                nation = set(nation)

                if len(numbers) > 0:
                    base_row["MOBILE_NUMBER"] = "/".join(numbers)
                if len(addresses) > 0:
                    base_row["Address"] = "/".join(addresses)
                if len(Date_of_birth) > 0:
                    base_row["DOB"] = "/".join(Date_of_birth)
                if len(nation) > 0:
                    base_row["NAT"] = "/".join(nation)

                filtered_df = filtered_df.append(base_row)
            elif group["SCORE"].iloc[0] > 2:
                filtered_df = filtered_df.append(group)

        print("finished loop")

        # filtered_df["SCORE"] = filtered_df.apply(score_record, axis=1)
        filtered_df["SCORE"] = [self.score_tuple(x) for x in
                                zip(filtered_df["FULL_NAME_ARA"], filtered_df["MOBILE_NUMBER"], filtered_df["NAT"],
                                    filtered_df["Address"], filtered_df["DOB"])]

        # state.df = filtered_df
        filtered_df = filtered_df.sort_values(['ROW_ID'],ascending=True)
        filtered_df.to_csv("./data/saved_ds_fltr.csv")

    def save_df(self,chosen_csv):
        chosen_csv = chosen_csv.replace(".csv", "").split("(")[0]

        i = 1
        while os.path.exists(f"{chosen_csv}({i}).csv"):
            i += 1
        new_path = f"{chosen_csv}({i}).csv"
        state.df.to_csv(new_path)
        st.success(f"New dataframe is saved  in {new_path}")


    def get(self):

        self.df = self.reformat_df("./data/ds.csv")
        print("verifing cells...")
        self.verify_cells()
        print("cells verified.")

        print("Filling records...")

        self.fill_records()

        print("records filled.")


        return {"df_after_len":len(self.df) }

class Golden_Record_load(Resource):
    def get(self):
        df = pd.Dataframe()


class Golden_Record_stats(Resource):

    stats_dict = {}

    def calculate_age(self,row):
        try:
            day, month, year = row["DOB"].split("/")
            year = int(year)
            month = int(month)
            day = int(day)

            today = date.today()

            return today.year - year - ((today.month, today.day) < (month, day))
        except:
            return 0

    def calculate_age_list(self,dob):
        try:

            day, month, year = dob.split("/")
            year = int(year)
            month = int(month)
            day = int(day)

            today = date.today()

            return today.year - year - ((today.month, today.day) < (month, day))
        except:
            return 0

    def generate_json(self,df_file):
        path = "./data"
        chosen_csv = os.path.join(path, df_file)
        df = pd.read_csv(chosen_csv)
        df_dict = df.to_dict()
        df_json = jsonify(df_dict)

        print(df_json)
        return df_json

    def get_Statistics(self,df: pd.DataFrame,df_file):
        # df["age"] = df.apply(calculate_age, axis=1)
        # nationality_dist = df["NAT"].value_counts()
        # duplicate_persons_dist = df["ID_NUMBER"].value_counts()
        # age_dist = df["age"].value_counts()
        self.generate_json(df_file)
        null_nat = len(df[(df['NAT'] == "") | (df['NAT'] == "nan" )]['NAT'])
        null_dob = len(df[(df['DOB'] == "" ) | (df['DOB'] == "nan" )]['DOB'])
        null_ID = len(df[df['ID_NUMBER'] == ""]['ID_NUMBER'])
        duplication_ratio = (
            df[df.duplicated(subset=["ID_NUMBER", "FULL_NAME_ARA"], keep=False)].shape[0]
            # * 100
            # / df.shape[0]
        )

        duplication_ID = df[df.duplicated(subset=["ID_NUMBER"], keep=False)].shape[0]

        number_of_rows = len(df["ID_NUMBER"])
        print("number of rows2: " + str(number_of_rows ))

        return {
            # "Nationality distribution": nationality_dist,
            # "Duplicate person distribution": duplicate_persons_dist,
            # "Age distribution": age_dist,
            "nnat":null_nat, #"Number of empty nationality": null_nat,
            "ndob":null_dob,#"Number of empty date of birth": null_dob,
            "nId":null_ID,#"Number of empty ID": null_ID,
            "dIdNmae":duplication_ratio,#"Duplication based on ID and Name": duplication_ratio,
            "dId":duplication_ID,#"Number of duplicated IDs": duplication_ID,
            "nrecord":number_of_rows#"Number of records": number_of_rows,
        }
    def reformat_df(self,chosen_csv):
        df = pd.read_csv(chosen_csv, index_col=0)

        df.to_csv("./data/ds_check0.csv")

        df["ID_NUMBER"] = df["ID_NUMBER"].fillna(0)

        df['DOB'] = df['DOB'].str.strip()
        df["DOB"] = df["DOB"].fillna('')
        df["NAT"] = df["NAT"].astype("str", errors="ignore")
        df['NAT'] = df['NAT'].str.strip()
        df["NAT"] = df["NAT"].fillna('')
        df["NAT"] = df["NAT"].replace("nan", "")
        df['ID_NUMBER'] = df['ID_NUMBER'].astype(np.int64)
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str", errors="ignore")
        df['ID_NUMBER'] = df['ID_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df['MOBILE_NUMBER'] = df['MOBILE_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].fillna('')
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace("0.0", "").replace("nan", "")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace(".0", "").replace("nan", "")
        df["PDF_PATH"] = df["PDF_PATH"].astype("str", errors="ignore").replace("nan", "")
        df["ID_PATH"] = df["ID_PATH"].astype("str", errors="ignore").replace("nan", "")
        df = df.fillna("")

        print("empty nat = " + str(len(df[df['NAT'] == ""]['NAT'])))

        df = df.reset_index(drop=True)

        return df

    def write(self,df_file):

        path = "./data"
        # extension = "csv"
        # csvs = glob.glob(f"{path}/**/*.{extension}", recursive=True)
        print('file name: ' + df_file)
        chosen_csv = os.path.join(path,df_file)

        # chosen_csv = os.path.join(path, df_file)
        df = self.reformat_df(chosen_csv)

        df.to_csv("./data/ds_check.csv")
        # df = pd.read_csv(chosen_csv, index_col=0)
        # df = pd.read_csv("./data/saved_ds_fltr.csv", encoding='utf-8')
        number_of_rows = len(df)
        print("number of rows1: " + str(number_of_rows))
        df = df.sort_values(by='ID_NUMBER').reset_index()
        # df = df[:20000]
        df = df.reset_index(drop=True)

        df.to_json(r"C:\jrc\test1.json")

        columns = df.columns.tolist()
        age_list = df['DOB']
        age_list = [self.calculate_age_list(x) for x in age_list]
        df["age"] = age_list
        output_path = "/".join(chosen_csv.split("/")[:-1])

        csv_name = chosen_csv.split("/")[-1].replace(".csv", "")
        report_path = os.path.join(output_path, f"{csv_name}_report.html")
        profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True, minimal=True)

        print("after profiling")

        report_path = "output.html"
        profile.to_file(report_path)

        print(report_path)
        abs_path = "file://" + os.path.abspath(report_path)
        print(abs_path)
        # webbrowser.open(abs_path)
        stats = self.get_Statistics(df,df_file)

        for key, value in stats.items():
            if "distribution" in key:
                fig, ax = plt.subplots()
                ax.hist(value, bins=20, align="mid")
                ax.set_title(key)
                ax.set_xlabel("value")
                ax.set_ylabel("Count")
                print("saving plot to " + os.path.join(path,'stats_plot' + str(key) + '.png'))
                plt.savefig(os.path.join(path,'stats_plot' + str(key) + '.png'))

            else:
                self.stats_dict[key] = value


    def get(self,df_file):

        self.write(df_file)

        return jsonify(self.stats_dict)

class Golden_Record_Docs(Resource):

    saved_df = pd.DataFrame()
    def reformat_df(self,chosen_csv):
        df = pd.read_csv(chosen_csv, index_col=0)


        df["ID_NUMBER"] = df["ID_NUMBER"].fillna(0)

        df['DOB'] = df['DOB'].str.strip()
        df["DOB"] = df["DOB"].fillna('')
        df["NAT"] = df["NAT"].astype("str", errors="ignore")
        df['NAT'] = df['NAT'].str.strip()
        df["NAT"] = df["NAT"].fillna('')
        df["NAT"] = df["NAT"].replace("nan", "")
        df['ID_NUMBER'] = df['ID_NUMBER'].astype(np.int64)
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str", errors="ignore")
        df['ID_NUMBER'] = df['ID_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df['MOBILE_NUMBER'] = df['MOBILE_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].fillna('')
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace("0.0", "").replace("nan", "")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace(".0", "").replace("nan", "")
        df["PDF_PATH"] = df["PDF_PATH"].astype("str", errors="ignore").replace("nan", "")
        df["ID_PATH"] = df["ID_PATH"].astype("str", errors="ignore").replace("nan", "")
        df = df.fillna("")

        df = df.reset_index(drop=True)

        return df

    def score_tuple(self,rec_tuple):  # (FULL_NAME_ARA,MOBILE_NUMBER,NAT,Address,DOB)
        score = 0
        if rec_tuple[0] != "" and len(rec_tuple[0]) >= 3:
            score += 10
        if rec_tuple[1] != "" and len(rec_tuple[1]) >= 3:
            score += 2
        if rec_tuple[2] != "" and len(rec_tuple[2]) >= 3:
            score += 5
        if rec_tuple[3] != "" and len(rec_tuple[3]) >= 3:
            score += 2
        if rec_tuple[4] != "" and len(rec_tuple[4]) >= 3:
            score += 5

        return score

    def get_pdf_dict(self,pdf_dir: str):
        files = glob.glob(f"{pdf_dir}*.pdf")
        dict_path = os.path.join(pdf_dir, "record_name_dict.json")

        if os.path.isfile(dict_path):
            with open(dict_path) as f:
                pdf_dict = json.load(f)
        else:
            pdf_dict = {}

        if len(files) == 0:
            print("The directory you chose does not contain any document with PDF format")
        else:
            imgs_dir = os.path.join(pdf_dir, "pdf_imgs")
            if not os.path.isdir(imgs_dir):
                os.mkdir(imgs_dir)

            for pdf_path in tqdm(files):
                pages = convert_from_path(pdf_path, dpi=500)
                file_name = os.path.basename(pdf_path).split(".")[0]
                page = pages[0]
                img_path = os.path.join(imgs_dir, f"{file_name}.jpg")
                page.save(img_path, "JPEG")

                name = read_name_img(img_path=img_path)
                id = read_id_img(img_path=img_path)
                dob = read_dob_img(img_path=img_path)
                nationality = read_nationality_img(img_path=img_path)

                file_info = {
                    "name": name,
                    "dob": dob,
                    "nationality": nationality,
                    "file_path": pdf_path,
                }

                if id in pdf_dict.keys():
                    pdf_dict[id].append(file_info)
                else:
                    pdf_dict[id] = [file_info]

            pdf_dict = {
                x: sorted(
                    pdf_dict[x],
                    reverse=True,
                    key=lambda x: len(list(filter(None, x.values()))),
                )
                for x in pdf_dict.keys()
            }

            with open(dict_path, "w") as f:
                json.dump(pdf_dict, f)
            print(pdf_dict)
            print(f"finished processing, record attachment reference exists in {dict_path}")

    # @st.cache(suppress_st_warning=True)
    def get_img_dict(self,img_dir):
        extensions = ["png", "jpeg", "jpg"]
        files = []
        test_files = []
        for ext in extensions:
            files.extend(glob.glob(f"{img_dir}"+"\\**\\**" + f".{ext}"))
            # test_files.extend(glob.glob(f"{img_dir}"+"\\**\\**"))

        if len(files) == 0:
            print("The directory you chose does not contain any document with images format")

        else:
            for img_path in tqdm(files):
                print(img_path)
            dict_path = os.path.join(img_dir, "record_name_dict.json")
            if os.path.isfile(dict_path):
                with open(dict_path) as f:
                    img_dict = json.load(f)
            else:
                img_dict = {}

            for img_path in tqdm(files):
                name = read_name_img(img_path=img_path)
                print("image path: " + img_path)
                print("after reading name from image")
                print(name)
                id = read_id_img(img_path=img_path)
                dob = read_dob_img(img_path=img_path)
                nationality = read_nationality_img(img_path=img_path)

                file_info = {
                    "name": name,
                    "dob": dob,
                    "nationality": nationality,
                    "file_path": img_path,
                }

                if id in img_dict.keys():

                    img_dict[id].append(file_info)
                else:
                    print("id = " + str(id))
                    img_dict[id] = [file_info]

            # sort the list of files based on the amount of information in them
            img_dict = {
                x: [sorted(
                    img_dict[x],
                    reverse=True,
                    key=lambda x: len(list(filter(None, x.values()))),
                )]
                for x in img_dict.keys()
            }

            with open(dict_path, "w") as f:
                json.dump(img_dict, f)
            print(img_dict)
            print(
                f"finished processing, record attachment reference exists in {dict_path}"
            )

    # @st.cache(suppress_st_warning=True)
    def clean_documents(self,chosen_dir):
        dict_path = os.path.join(chosen_dir, "record_name_dict.json")

        duplicates_path = os.path.join(chosen_dir, "duplicates")
        if not os.path.isdir(duplicates_path):
            os.mkdir(duplicates_path)

        norecognition_path = os.path.join(chosen_dir, "no_recognition")
        if not os.path.isdir(norecognition_path):
            os.mkdir(norecognition_path)

        if os.path.isfile(dict_path):
            with open(dict_path) as f:
                doc_dict = json.load(f)
        else:
            return "no dict"


        null_files = doc_dict.get("null")
        if null_files:
            for null_file in tqdm(null_files):
                null_path = null_file["file_path"]
                file_name = os.path.basename(null_path)
                new_path = os.path.join(norecognition_path, file_name)
                os.replace(null_path, new_path)
            del doc_dict["null"]



        for key in tqdm(doc_dict.keys()):
            imgs = doc_dict[key]
            for image in imgs[1:]:
                img_path = image["file_path"]
                file_name = os.path.basename(img_path)
                new_path = os.path.join(duplicates_path, file_name)
                os.replace(img_path, new_path)
            doc_dict[key] = doc_dict[key][0]

        with open(dict_path, "w") as f:
            json.dump(doc_dict, f)


        return "success"

    def attach_records(self,dict_path, chosen_csv, chosen_format, df):
        # df["FULL_NAME_ARA"] = df["FULL_NAME_ARA"].astype(str)
        with open(dict_path) as f:
            record_dict = json.load(f)
        print("rec dict : " + str(record_dict) )
        df["NAT"] = df["NAT"].astype(str)
        for index, row in df.iterrows():
            record_id = df.at[index, "ID_NUMBER"]
            record_name = df.at[index, "FULL_NAME_ARA"]
            record_DOB = df.at[index, "DOB"]
            record_NAT = df.at[index, "NAT"]

            file_info = record_dict.get(record_id)
            if file_info:
                if file_info[0].get("name") != record_name:
                    df.at[index, "FULL_NAME_ARA"] = file_info[0].get("name")
                print("len of record dob " + str(len(str(record_DOB))))
                if (str(record_DOB) == "nan" or str(record_DOB) == "" ) and file_info[0].get("dob"):
                    print("record dob = " + str(record_DOB))
                    df.at[index, "DOB"] = datetime.datetime(int(file_info[0].get("dob").get("y")),int(file_info[0].get("dob").get("m")),int(file_info[0].get("dob").get("d"))).date().strftime('%d/%m/%Y')

                if ( str(record_NAT) == "nan" or str(record_DOB) == "" ) and file_info[0].get("nationality"):
                    print(file_info[0])
                    df.at[index, "NAT"] = file_info[0].get("nationality")

                if chosen_format == "PDFs":
                    df.at[index, "PDF_PATH"] = file_info[0].get("file_path")
                else:
                    df.at[index, "ID_PATH"] = file_info[0].get("file_path")
        # df.to_csv(chosen_csv)

        df["SCORE"] = [self.score_tuple(x) for x in
                                  zip(df["FULL_NAME_ARA"], df["MOBILE_NUMBER"],
                                      df["NAT"], df["Address"], df["DOB"])]

        csv_name = chosen_csv.split("/")[-1].replace(".csv", "")
        output_path = "/".join(chosen_csv.split("/")[:-1])
        new_csv_name = os.path.join(output_path, f"{csv_name}_attached_docs.csv")
        df.to_csv(new_csv_name)

        self.saved_df = df

    def write(self):

        chosen_format = "Images"

        chosen_dir = "./data/jpgs"
        chosen_csv = "./data/saved_ds_fltr.csv"
        df = self.reformat_df(chosen_csv)
        # df = pd.read_csv(chosen_csv, index_col=0)
        # df = df[:20000]
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df["PDF_PATH"] = df["PDF_PATH"].astype("str", errors="ignore")
        df["ID_PATH"] = df["ID_PATH"].astype("str", errors="ignore")

        df = df.reset_index(drop=True)

        # Process Directory

        self.get_img_dict(chosen_dir)

        # Clean Documents

        print(chosen_dir)
        out = self.clean_documents(chosen_dir)
        if out == "no dict":
            print("The directory you chose does not contain The name_record json file")
        elif out == "success":
            print("finished processing, record attachment reference has been updated")

        # Attach records with files
        dict_path = os.path.join(chosen_dir, "record_name_dict.json")
        self.attach_records(dict_path, chosen_csv, chosen_format, df)
        print("data frame is updated")


    def get(self):
        self.write()
        return jsonify({"length":len(self.saved_df)})

class Basic_Data_Json(Resource):

    def get_df_json(self,df_file):
        # print('test 2')
        # path = "./data"
        # chosen_csv = os.path.join(path, 'ds.csv')
        df2 = self.reformat_df(df_file)

        df2 = df2[:30]

        print("prepared df2")
        print(df2.columns)
        df_dict = df2.to_dict()
        df_json = jsonify(df_dict)

        return df_json

    def reformat_df(self,chosen_csv):
        df = pd.read_csv(chosen_csv, index_col=0)

        df["ID_NUMBER"] = df["ID_NUMBER"].fillna(0)

        df['DOB'] = df['DOB'].str.strip()
        df["DOB"] = df["DOB"].fillna('')
        df["NAT"] = df["NAT"].astype("str", errors="ignore")
        df['NAT'] = df['NAT'].str.strip()
        df["NAT"] = df["NAT"].fillna('')
        df["NAT"] = df["NAT"].replace("nan", "")
        df['ID_NUMBER'] = df['ID_NUMBER'].astype(np.int64)
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str", errors="ignore")
        df['ID_NUMBER'] = df['ID_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df['MOBILE_NUMBER'] = df['MOBILE_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].fillna('')
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace("0.0", "").replace("nan", "")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace(".0", "").replace("nan", "")
        df["PDF_PATH"] = df["PDF_PATH"].astype("str", errors="ignore").replace("nan", "")
        df["ID_PATH"] = df["ID_PATH"].astype("str", errors="ignore").replace("nan", "")
        df = df.fillna("")

        df = df.reset_index(drop=True)

        return df

    def get(self):
        path = "./data"
        chosen_csv = os.path.join(path, 'ds.csv')
        return self.get_df_json(chosen_csv)

class Filtered_Data_Json(Resource):

    def get_df_json(self,df_file):
        # print('test 2')
        # path = "./data"
        # chosen_csv = os.path.join(path, 'ds.csv')
        df2 = self.reformat_df(df_file)

        df2 = df2[:20]

        df_dict = df2.to_dict()
        df_json = jsonify(df_dict)

        return df_json

    def reformat_df(self,chosen_csv):
        print("start reading CSV!")
        df = pd.read_csv(chosen_csv, index_col=0)

        df["ID_NUMBER"] = df["ID_NUMBER"].fillna(0)

        df['DOB'] = df['DOB'].str.strip()
        df["DOB"] = df["DOB"].fillna('')
        df["NAT"] = df["NAT"].astype("str", errors="ignore")
        df['NAT'] = df['NAT'].str.strip()
        df["NAT"] = df["NAT"].fillna('')
        df["NAT"] = df["NAT"].replace("nan", "")
        df['ID_NUMBER'] = df['ID_NUMBER'].astype(np.int64)
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str", errors="ignore")
        df['ID_NUMBER'] = df['ID_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df['MOBILE_NUMBER'] = df['MOBILE_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].fillna('')
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace("0.0", "").replace("nan", "")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace(".0", "").replace("nan", "")
        df["PDF_PATH"] = df["PDF_PATH"].astype("str", errors="ignore").replace("nan", "")
        df["ID_PATH"] = df["ID_PATH"].astype("str", errors="ignore").replace("nan", "")
        df = df.fillna("")

        df = df.reset_index(drop=True)

        return df

    def delete_cols_from_df(self,chosen_csv):
        print("start reading CSV!")
        df = pd.read_csv(chosen_csv, index_col=0)

        df["ID_NUMBER"] = df["ID_NUMBER"].fillna(0)

        df['DOB'] = df['DOB'].str.strip()
        df["DOB"] = df["DOB"].fillna('')
        df["NAT"] = df["NAT"].astype("str", errors="ignore")
        df['NAT'] = df['NAT'].str.strip()
        df["NAT"] = df["NAT"].fillna('')
        df["NAT"] = df["NAT"].replace("nan", "")
        df['ID_NUMBER'] = df['ID_NUMBER'].astype(np.int64)
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str", errors="ignore")
        df['ID_NUMBER'] = df['ID_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df.drop(columns=['MOBILE_NUMBER'], inplace=True)
        
        df = df.fillna("")

        df = df.reset_index(drop=True)

        return df
        
    def get(self):
        path = "./data"
        chosen_csv = os.path.join(path, 'saved_ds_fltr.csv')
        print("filtered data retrieve 1")
        return self.get_df_json(chosen_csv)

class Docs_Data_Json(Resource):

    def get_df_json(self,df_file):
        # print('test 2')
        # path = "./data"
        # chosen_csv = os.path.join(path, 'ds.csv')
        df2 = self.reformat_df(df_file)

        df2 = df2[:20]

        df_dict = df2.to_dict()
        df_json = jsonify(df_dict)

        return df_json

    def reformat_df(self,chosen_csv):
        print("start reading CSV!")
        df = pd.read_csv(chosen_csv, index_col=0)

        df["ID_NUMBER"] = df["ID_NUMBER"].fillna(0)

        df['DOB'] = df['DOB'].str.strip()
        df["DOB"] = df["DOB"].fillna('')
        df["NAT"] = df["NAT"].astype("str", errors="ignore")
        df['NAT'] = df['NAT'].str.strip()
        df["NAT"] = df["NAT"].fillna('')
        df["NAT"] = df["NAT"].replace("nan", "")
        df['ID_NUMBER'] = df['ID_NUMBER'].astype(np.int64)
        df["ID_NUMBER"] = df["ID_NUMBER"].astype("str", errors="ignore")
        df['ID_NUMBER'] = df['ID_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df['MOBILE_NUMBER'] = df['MOBILE_NUMBER'].str.strip()
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].fillna('')
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].astype("str", errors="ignore")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace("0.0", "").replace("nan", "")
        df["MOBILE_NUMBER"] = df["MOBILE_NUMBER"].replace(".0", "").replace("nan", "")
        df["PDF_PATH"] = df["PDF_PATH"].astype("str", errors="ignore").replace("nan", "")
        df["ID_PATH"] = df["ID_PATH"].astype("str", errors="ignore").replace("nan", "")
        df = df.fillna("")

        df = df.reset_index(drop=True)

        return df

    def get(self):
        path = "./data"
        chosen_csv = os.path.join(path, 'saved_ds_fltr_attached_docs.csv')
        return self.get_df_json(chosen_csv)



api.add_resource(Golden_Record_stats, "/ds_stats/<df_file>")
api.add_resource(Golden_Record, "/ds_filter/")
api.add_resource(Golden_Record_Docs, "/ds_docs/")
api.add_resource(Basic_Data_Json, "/ds_json/")
api.add_resource(Filtered_Data_Json, "/ds_filtered_json/")
api.add_resource(Docs_Data_Json, "/ds_docs_json/")


if __name__ == "__main__":
    app.run(debug=True)









#with open('plty_div.txt', 'w') as f:
 #   f.write(plt_div)


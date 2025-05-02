import functions
import data_process as answers


def main(file_Path):
    df = functions.read_csv_files(file_Path)
    answers.answer_1(df)
    answers.answer_2(df)
    answers.answer_3(df)
 
if __name__ == "__main__":
    file_path = './files/dataset.csv'
    main(file_path)
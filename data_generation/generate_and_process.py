from data_generation.generate_data import generate_data
from data_generation.process_data import process_data

if __name__ == "__main__":

    i = 0
    while True:
        if (generate_data(seed=i, prompt_accept=True)):
            process_data()
        i += 1

import string
import random
import csv

def generate_random_string():
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for i in range(10))  # 10 characters long

def main():
    gibberish_list = [generate_random_string() for _ in range(100)]  # Generate 100 random strings

    with open(r'E:\document\MIA\gibberish.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[g] for g in gibberish_list])

if __name__ == "__main__":
    main()

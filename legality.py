def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.split()
            string = (line[0], line[1])
            data.append(string)
    return data


def filter_binary_numbers0(binary_numbers):
    filtered_numbers = []
    unFiltered_numbers = []
    for binary in binary_numbers:
        count_ones = binary[0].count('1')
        if count_ones >= 8 and count_ones <= 12:
            filtered_numbers.append(binary[1])
        else:
            unFiltered_numbers.append(binary[1])
    return filtered_numbers, unFiltered_numbers

def filter_binary_numbers1(binary_numbers):
    filtered_numbers = []
    unFiltered_numbers = []
    for binary in binary_numbers:
        count_ones = binary[0].count('1')
        if count_ones < 8:
            filtered_numbers.append(binary[1])
        else:
            unFiltered_numbers.append(binary[1])
    return filtered_numbers, unFiltered_numbers

def main_nn0():
    print("nn0")
    data = load_data("nn0.txt")
    filtered_numbers, unFiltered_numbers = filter_binary_numbers0(data)
    #print("filtered_numbers", filtered_numbers)
    #print("unFiltered_numbers", unFiltered_numbers)
    count_zero_filtered_numbers = filtered_numbers.count('0')
    count_ones_unFiltered_numbers = unFiltered_numbers.count('1')
    print("count_zero_filtered_numbers",count_zero_filtered_numbers)
    print("count_ones_unFiltered_numbers", count_ones_unFiltered_numbers)

def main_nn1():
    print("nn1")
    data = load_data("nn1.txt")
    filtered_numbers, unFiltered_numbers = filter_binary_numbers1(data)
    #print("filtered_numbers", filtered_numbers)
    #print("unFiltered_numbers", unFiltered_numbers)
    count_zero_filtered_numbers = filtered_numbers.count('0')
    count_ones_unFiltered_numbers = unFiltered_numbers.count('1')
    print("count_zero_filtered_numbers",count_zero_filtered_numbers)
    print("count_ones_unFiltered_numbers", count_ones_unFiltered_numbers)

if __name__ == '__main__':
    main_nn0()
    main_nn1()


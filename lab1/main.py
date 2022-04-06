def data_headers(data):
    return data[0].split(',')[1:]


def process_data(data):
    countries = []
    for row in data[1:]:
        country = row.split(',')
        countries.append(dict(
            name=str(country[0]),
            capital=str(country[1]),
            population=str(country[2]),
            area=str(country[3]),
            gdp=str(country[4]),
            country_code=str(country[5]),
            phone_code=str(country[6]),
            currency=str(country[7]),
            leader=str(country[8]),
            domain=str(country[9]),
            hdi=str(country[10])
        ))
    return countries


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines


def main():
    raw_data = read_file('db.csv')
    headers = data_headers(raw_data)
    countries = process_data(raw_data)

    matched_countries = dict()
    while len(matched_countries) != 1 and len(headers) != 0:

        search_criteria = dict()
        for header in headers:
            parameter = input(f'Enter parameter "{header}" or leave it empty: ')
            if parameter:
                search_criteria[header] = parameter
        matched_countries = dict()
        for country in countries:
            for k, v in search_criteria.items():
                if country[k] == v:
                    matched_countries[country['name']] = country

        if len(matched_countries) != 1:
            print(f'{len(matched_countries)} countries match parameters\n{search_criteria}. Please clarify your search\n')

    print(f'Found country {list(matched_countries.keys())[0]}')


if __name__ == '__main__':
    main()

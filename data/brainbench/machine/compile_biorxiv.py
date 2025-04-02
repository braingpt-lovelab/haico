import requests
import json


def download_abstracts(start_date, end_date, n_cursor, output_format):
    base_url = 'https://api.biorxiv.org/details/biorxiv/{}/{}/{}'
    interval = f'{start_date}/{end_date}'
    all_abstracts = []

    for cursor in range(0, n_cursor):
        url = base_url.format(interval, cursor, output_format)
        print(f"Downloading from {url}, {cursor}th cursor")

        response = requests.get(url)
        data = response.json()

        messages = data.get('messages', [])
        for message in messages:
            print(message)

        abstracts = data.get('collection', [])
        print(f"Found {len(abstracts)} abstracts")

        for abstract in abstracts:
            abstract_content = abstract['abstract']
            all_abstracts.append({"text": abstract_content})
    
    # Save all abstracts to one JSON file
    with open(datafile_path, 'w', encoding='utf-8') as f:
        json.dump(all_abstracts, f, ensure_ascii=False, indent=2)
    print(f"Saved all abstracts to {datafile_path}")


if __name__ == "__main__":
    start_date = '2021-06-01'
    end_date = '2021-12-01'
    n_samples = 1000
    biorxiv_fixed_n_samples_per_request = 100
    n_cursor = n_samples // biorxiv_fixed_n_samples_per_request
    datafile_path = f"data/biorxiv_abstracts_{start_date}_{end_date}.json"
    output_format = 'json'

    download_abstracts(
        start_date, 
        end_date, 
        n_cursor, 
        output_format
    )

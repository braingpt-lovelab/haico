import urllib.request
import xml.etree.ElementTree as ET
import json


def download_arxiv_abstracts(start_date, end_date, max_results, datafile_path):
    base_url = 'http://export.arxiv.org/api/query?'
    query_params = {
        'search_query': f'submittedDate:[{start_date} TO {end_date}]',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    encoded_params = urllib.parse.urlencode(query_params)
    full_url = base_url + encoded_params
    print(f"Downloading from {full_url}")

    data = urllib.request.urlopen(full_url)
    xml_data = data.read().decode('utf-8')
    root = ET.fromstring(xml_data)

    all_abstracts = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        all_abstracts.append({"text": summary})
    
    # Save all abstracts to one JSON file
    with open(datafile_path, 'w', encoding='utf-8') as f:
        json.dump(all_abstracts, f, ensure_ascii=False, indent=2)
    print(f"Saved all abstracts to {datafile_path}")


if __name__ == "__main__":
    start_date = '20210601'
    end_date = '20211231'
    max_results = 1000
    datafile_path = f"data/arxiv_abstracts_{start_date}_{end_date}.json"
    
    download_arxiv_abstracts(start_date, end_date, max_results, datafile_path)

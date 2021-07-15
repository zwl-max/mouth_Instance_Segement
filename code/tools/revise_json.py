import json

if __name__ == "__main__":
    json_data = json.load(open('submit-6-16.segm.json', 'r'))
    for i, data in enumerate(json_data):
        if data['category_id'] == 3:
            json_data[i]['category_id'] = 5
        if data['category_id'] == 5:
            json_data[i]['category_id'] = 3
    json.dump(json_data, open('revised-submit-6-16.segm.json', 'w'))
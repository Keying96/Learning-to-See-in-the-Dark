# -*- coding:utf-8 -*-
import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)



if __name__ == "__main__":

    # print('Dowloading Sony subset... (25GB)')
    # download_file_from_google_drive('10kpAcvldtcb9G2ze5hTcF1odzu4V_Zvh', 'dataset/Sony.zip')
    #
    # print('Dowloading Fuji subset... (52GB)')
    # download_file_from_google_drive('12hvKCjwuilKTZPe9EZ7ZTb-azOmUA3HT', 'dataset/Fuji.zip')
    #
    # os.system('unzip dataset/Sony.zip -d dataset')
    # os.system('unzip dataset/Fuji.zip -d dataset')

    #设置第二个参数，对应到drive中需要下载的文件夹
    # print('Dowloading Iphone subset... ')
    # download_file_from_google_drive('101eXwidf9a5ZmvAt__sLWI9H9g_3KMp7', 'Iphone.zip')
    # print(os.path.abspath('.'))
    # os.system('unzip ./Iphone.zip -d dataset')
    print('Dowloading Iphone subset... ')
    download_file_from_google_drive('1eCBEgnzmKO4zhdsq7G1IEap0G2r7cJHK', 'UnderexposedImage.zip')
    print(os.path.abspath('.'))
    # os.system('unzip ./Iphone.zip -d dataset')

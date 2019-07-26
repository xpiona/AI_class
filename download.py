# code.py
# Link:https://code.i-harness.com/ko/q/6e87e6
from requests import get  # to make GET request

def download(url, file_name):
    with open(file_name, "wb") as file:   # open in binary mode
        response = get(url)               # get request
        file.write(response.content)      # write to file

if __name__ == '__main__':
	url = "http://wasabisyrup.com/storage/gallery/-quJruo2cWQ/m0071_6lMClr7aldA.jpg"
	download(url,"iml.jpg")

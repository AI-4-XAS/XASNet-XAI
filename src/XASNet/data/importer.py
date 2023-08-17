import os
import sys
import ssl
import urllib
import zipfile
import os.path as osp
from typing import *



def download_url(
    url: str, 
    folder: str, 
    filename: Optional[str] = None
    ):
    """function to download the qm9 data from url 

    Args:
        url (str): url where qm9.zip is located
        folder (str): folder to save the data
        filename (Optional[str], optional)

    Returns:
        str : path of the saved qm9.zip data
    """
    if filename is None:
        filename = url.rpartition('/')[2]

    path = osp.join(folder, filename)

      
    if osp.exists(path):
        print(f"using the existing file {filename}")
        return path
    
    contex = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=contex)

    with open(path, 'wb') as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
        f.write(chunk)

    return path

def extract_zip(path: str, folder:str):
    """extracting qm9.zip file

    Args:
        path (str): path to save the extracted data
        folder (str)
    """
    print(path)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)
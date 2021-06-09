# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 17:29:01 2021

@author: 11327
"""

import json

 # initial import calls
import argparse

parser = argparse.ArgumentParser( description ='HW02 Task1')
parser.add_argument('--subclass_list', nargs ='*',type =str,required = True )
parser.add_argument('--images_per_subclass', type =int, required = True )
parser.add_argument('--data_root', type =str , required = True )
parser.add_argument('--main_class',type =str , required =True )
parser.add_argument('--imagenet_info_json', type =str , required = True )
args , args_other = parser.parse_known_args()

'''
Reference : https://github.com/johancc/ImageNetDownloader
'''

import requests
from PIL import Image
import os

from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL

def get_image ( img_url , class_folder ):
    if len ( img_url ) <= 1:
        print('url is useless Do something')
    try:
        img_resp = requests.get( img_url , timeout = 1 )
    except ConnectionError :
        print('ConnectionError')
        return False
    except ReadTimeout :
        print('ReadTimeout')
        return False
    except TooManyRedirects :
        print('TooManyRedirects')
        return False
    except MissingSchema :
        print('MissingSchema')
        return False
    except InvalidURL :
        print('InvalidURL')
        return False
        
    if not 'content-type' in img_resp.headers :
        print('not content-type')
        return False
    if not 'image' in img_resp.headers['content-type']:
        print('not image')
        return False
    if (len ( img_resp.content ) < 1000 ):
        return False
    
    img_name = img_url.split ('/')[-1]
    img_name = img_name.split ("?")[0]
    
    if (len ( img_name ) <= 1 ):
        print('missing image name')
        return False
    if not 'flickr' in img_url :
        print('Missing non - flickr images are difficult to handle . Do something .')
        return False
        
    img_file_path = os.path.join( class_folder , img_name )
    
    # check if the file has already been downloaded
    if os.path.exists(img_file_path):
        return False
    
    with open ( img_file_path , 'wb') as img_f :
        img_f.write( img_resp.content )
    
    # Resize image to 64x64
    im = Image.open( img_file_path )
    
    if im.mode != "RGB":
        im = im.convert ( mode ="RGB")
    
    im_resized = im.resize (( 64 , 64 ),Image.BOX )
    # Overwrite original image with d
    im_resized.save(img_file_path)
    
    return 1

# create folder
folderpath = os.path.join('./'+args.data_root,args.main_class)
folderpath = folderpath.replace('\\','/')
if not os.path.exists(folderpath):
    os.makedirs(folderpath)

with open(args.imagenet_info_json) as f:
    jsondate = json.load(f)

for subclass_name in args.subclass_list:
    # Find the identifier related to the subclass
    for I, content in jsondate.items():
        for temp,content2 in content.items():
            if subclass_name == content2:
                identifier = I
    
    # create the URL
    identifier = str(identifier)
    the_list_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + identifier
   
    resp = requests.get ( the_list_url )
    urls = [url.decode('utf -8') for url in resp.content.splitlines ()]

    # get_img
    successcount = 0
    for url in urls:
        c = get_image(url, folderpath)
        if c == 1:
            successcount = successcount + 1
        if successcount >= args.images_per_subclass:
            break
     
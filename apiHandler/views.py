import time
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
import ee
import cv2
import requests
import numpy as np
import threading
import os
import re
from datetime import datetime
import threading

def wait_for_results(timestamp):
    result_path = os.path.join(file_dir, 'images-result', f'{timestamp}.txt')
    while not os.path.exists(result_path):
        time.sleep(5)

ee.Initialize()

@csrf_exempt
def index(request):
    return HttpResponse("Hello, world. You're at the apiHandler index.")

@csrf_exempt
def postImage(request):
    if request.method == "GET":
        return HttpResponse("Submit a POST request with an image")
    
    if request.method == "POST":
        try:
            image = request.FILES['image']
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            name = f'img_{timestamp}.png'
            img_path = os.path.join(file_dir, 'images', name)
            with open(img_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            threading.Thread(target=wait_for_results, args=(timestamp,)).start()
            return JsonResponse({"message": "Image uploaded successfully", "identifier": ('img_'+timestamp)}, safe=False)
        except Exception as e:
            return HttpResponse(str(e), status=500)

@csrf_exempt
def postCoordinates(request):
    if request.method == "GET":
        return HttpResponse("Submit a POST request with coordinates JSON object")
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            bl = data.get('bl')
            br = data.get('br')
            tl = data.get('tl')
            tr = data.get('tr')

            if not all([bl, br, tl, tr]):
                return HttpResponse("Invalid data provided", status=400)

            with open(os.path.join(file_dir, 'preferences.json'), 'r', encoding='utf-8') as f:
                prefs = json.loads(f.read())


            zoom = 21
            url = prefs['url']
            headers = prefs['headers']
            tile_size = int(prefs['tile_size'])
            channels = int(prefs['channels'])

            lat1, lon1 = float(tl['lat']), float(tl['lng'])
            lat2, lon2 = float(br['lat']), float(br['lng'])

            img = download_image(lat1, lon1, lat2, lon2, zoom, url, headers, tile_size, channels)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            name = f'img_{timestamp}.png'
            img_path = os.path.join(prefs['dir'], name)
            cv2.imwrite(img_path, img)

            return JsonResponse({"message": "Image created successfully", "image_path": img_path}, safe=False)

        except Exception as e:
            return HttpResponse(str(e), status=500)

@csrf_exempt
def fetchResults(request):
    if request.method == "POST":
        data = json.loads(request.body)
        identifier = data.get('identifier')

        if not identifier:
            return HttpResponse("Invalid data provided", status=400)

        result_path = os.path.join(file_dir, 'images-result', f'{identifier}.txt')
        if os.path.exists(result_path):
            with open(result_path, 'r') as file:
                results = file.read()
            return JsonResponse({"results": results})
        else:
            return JsonResponse({"message": "Results not ready yet"}, status=202)


def download_tile(url, headers, channels):
    response = requests.get(url, headers=headers)
    arr =  np.asarray(bytearray(response.content), dtype=np.uint8)
    
    if channels == 3:
        return cv2.imdecode(arr, 1)
    return cv2.imdecode(arr, -1)

# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_image(lat1: float, lon1: float, lat2: float, lon2: float,
    zoom: int, url: str, headers: dict, tile_size: int = 256, channels: int = 3) -> np.ndarray:

    scale = 1 << zoom

    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    def build_row(tile_y):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile = download_tile(url.format(x=tile_x, y=tile_y, z=zoom), headers, channels)

            if tile is not None:
                tl_rel_x = tile_x * tile_size - tl_pixel_x
                tl_rel_y = tile_y * tile_size - tl_pixel_y
                br_rel_x = tl_rel_x + tile_size
                br_rel_y = tl_rel_y + tile_size

                img_x_l = max(0, tl_rel_x)
                img_x_r = min(img_w + 1, br_rel_x)
                img_y_l = max(0, tl_rel_y)
                img_y_r = min(img_h + 1, br_rel_y)

                cr_x_l = max(0, -tl_rel_x)
                cr_x_r = tile_size + min(0, img_w - br_rel_x)
                cr_y_l = max(0, -tl_rel_y)
                cr_y_r = tile_size + min(0, img_h - br_rel_y)

                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]


    threads = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        thread = threading.Thread(target=build_row, args=[tile_y])
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    return img


def image_size(lat1: float, lon1: float, lat2: float,
    lon2: float, zoom: int, tile_size: int = 256):

    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y

file_dir = os.path.dirname(__file__)
prefs_path = os.path.join(file_dir, 'preferences.json')
default_prefs = {
        'url': 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        'tile_size': 256,
        'channels': 3,
        'dir': os.path.join(file_dir, 'images'),
        'headers': {
            'cache-control': 'max-age=0',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
        },
        'tl': '',
        'br': '',
        'zoom': ''
    }

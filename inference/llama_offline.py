import numpy as np
import cv2
import sys
import time
import os
import requests
import base64
from typing import Any, Dict
import random
import socket

def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str

def send_request(url: str, try_count = 10, **kwargs: Any) -> dict:
    response = {}
    for attempt in range(try_count):
        try:
            response = _send_request(url, **kwargs)
            break
        except Exception as e:
            if attempt == 9:
                print(e)
                exit()
            else:
                print(f"Error: {e}. Retrying in 10-20 seconds...")
                time.sleep(10 + random.random() * 10)

    return response


def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.05)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=50)
                # resp = requests.post(url, headers=headers, json=payload, timeout=5)
                if resp.status_code == 200:
                    result = resp.json()
                    break
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                if time.time() - start_time > 50:
                    raise Exception("Request timed out after 50 seconds")

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result


class LlamaClient:
    def __init__(self, port: int = 12181, IP = None):
        if IP == None:
            self.url = f"http://localhost:{port}/llama"
        else:
            self.url = f"http://{IP}:{port}/llama"

    def predict(self, prompt: str = ""):
        response = send_request(self.url, caption=prompt)
        return response

if __name__ == "__main__":

    llm_client = LlamaClient(12181, IP="115.25.142.41") # change the ip according to your computer

    while True:
        prompt = input("\nPlease answer your question, enter 'quit' to leave:")
        start = time.time()
        prompt = prompt.strip()
        if prompt == "quit":
            print("Assistant: Looking forward to see you again.")
            print("Ending the process......")
            break

        response = llm_client.predict(prompt)
        end = time.time()
        print("Assistant: ", response)
        print(f"Inference time: {end - start} s.")
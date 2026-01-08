import os
import re
import json
import time
import math
import base64
import io
import random
import tempfile
import traceback
import concurrent.futures
from typing import Any, Dict, Optional, Union, List, Tuple
from io import BytesIO

# 导入必要的库
import torch
import torch.nn.functional as F
import numpy as np
import requests
import urllib3
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import folder_paths
import comfy.model_management

# 尝试导入 pandas
try:
    import pandas as pd
except ImportError:
    pd = None
    print("⚠️ [MyNodes] Warning: 'pandas' library not found. Batch nodes may fail.")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("Loading Custom Nodes: Grsai & XZL Utility Suite...")

# ==============================================================================
# 全局工具与配置 (Global Utils & Config)
# ==============================================================================

class AnyType(str):
    def __ne__(self, __value): return False

GLOBAL_SESSION = requests.Session()
_retry = Retry(total=0, connect=1, read=0, backoff_factor=1, status_forcelist=[500, 502, 503, 504], allowed_methods=frozenset(["GET", "POST"]))
_adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=_retry)
GLOBAL_SESSION.mount("https://", _adapter)
GLOBAL_SESSION.mount("http://", _adapter)
GLOBAL_HEADERS = {"User-Agent": "ComfyUI-Nkxx/5.7-StrictBypass"}

DEFAULT_GRSAI_KEY = os.environ.get("GRSAI_KEY_DEFAULT", "").strip()

SUPPORTED_MODELS = [
    "nano-banana-pro", "nano-banana-fast", "nano-banana-pro-vt",
    "nano-banana-2-lite", "nano-banana-2-pro", "nano-banana-pro-vip",
    "nano-banana-pro-4k-vip", "gemini-3-pro-image-preview",
    "seedream-4.5", "flux-pro-1.1", "gpt-image-1.5"
]
SUPPORTED_RESOLUTIONS = ["1K", "2K", "4K"]
SUPPORTED_ASPECT_RATIOS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]

def get_grsai_api_key(inline_key: str = "") -> str:
    if inline_key and inline_key.strip(): return inline_key.strip()
    if os.getenv("GRSAI_KEY", "").strip(): return os.getenv("GRSAI_KEY").strip()
    if DEFAULT_GRSAI_KEY: return DEFAULT_GRSAI_KEY
    return ""

def format_proxies(proxy_url: str) -> Optional[Dict[str, str]]:
    if not proxy_url or not proxy_url.strip(): return None
    p = proxy_url.strip()
    return {"http": p, "https": p}

def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    if not isinstance(tensor, torch.Tensor): return []
    images = []
    for i in range(tensor.shape[0]):
        img_np = (torch.clamp(tensor[i], 0, 1).cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA'))
    return images

def pil_to_tensor(pil_images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if not isinstance(pil_images, list): pil_images = [pil_images]
    tensors = []
    for pil_image in pil_images:
        arr = np.array(pil_image).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr)[None, ...])
    if not tensors: return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    return torch.cat(tensors, dim=0)

def safe_pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == 'RGBA':
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    if image.mode != 'RGB': return image.convert('RGB')
    return image

def download_image_robust(url: str, timeout: int = 60, proxies: Optional[Dict] = None) -> Image.Image:
    last_err = None
    for attempt in range(3):
        try:
            safe_timeout = max(15.0, float(timeout))
            resp = GLOBAL_SESSION.get(url, headers=GLOBAL_HEADERS, timeout=safe_timeout, proxies=proxies)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise Exception(f"DL Fail: {str(last_err)}")

def is_black_512_image(image: Image.Image) -> bool:
    if image is None: return False
    if image.size != (512, 512): return False
    try:
        gray = image.convert("L")
        min_val, max_val = gray.getextrema()
        if max_val < 10: return True
    except: pass
    return False

def calculate_dimensions(resolution: str, aspect_ratio: str) -> Tuple[int, int]:
    base_pixels = 1024 * 1024 
    if resolution == "2K": base_pixels = 2048 * 2048
    elif resolution == "4K": base_pixels = 3840 * 2160
    
    ratio_map = {
        "1:1": 1.0, "16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "3:4": 3/4,
        "3:2": 3/2, "2:3": 2/3, "21:9": 21/9, "5:4": 5/4, "4:5": 4/5, "auto": 1.0 
    }
    ratio = ratio_map.get(aspect_ratio, 1.0)
    width = int((base_pixels * ratio) ** 0.5)
    height = int(base_pixels / width)
    return ((width // 64) * 64, (height // 64) * 64)

def int_to_zh(n: int) -> str:
    if n == 0: return "零"
    chars = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    units = ["", "十", "百"]
    if n == 100: return "一百"
    s = str(n)
    length = len(s)
    result = []
    for i, digit in enumerate(s):
        val = int(digit)
        if val != 0:
            result.append(chars[val])
            result.append(units[length - 1 - i])
        else:
            if result and result[-1] != "零" and (length - 1 - i) > 0:
                result.append("零")
    final_str = "".join(result)
    if 10 <= n < 20 and final_str.startswith("一十"):
        final_str = final_str[1:]
    return final_str

# ==============================================================================
# API 客户端类
# ==============================================================================

def get_upload_token_zh(api_key: str, data: Optional[Dict] = None, proxies: Optional[Dict] = None) -> Dict:
    url = "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = GLOBAL_SESSION.post(url=url, headers=headers, json=data or {}, timeout=30, proxies=proxies)
    resp.raise_for_status()
    return resp.json()

def upload_file_zh(file_path: str = "", proxies: Optional[Dict] = None, specific_key: str = None) -> str:
    api_key = specific_key or os.getenv("GRSAI_KEY", "").strip()
    if not file_path or not api_key: return f"Error: Missing File or Key"
    if not os.path.exists(file_path): return f"Error: File not found {file_path}"
    ext = os.path.splitext(file_path)[1].lstrip(".") or "png"
    try:
        res = get_upload_token_zh(api_key, {"sux": ext}, proxies=proxies)
        if "data" not in res or "token" not in res["data"]: return f"Error: Get Token Failed - {res}"
        token, key, url, domain = (res["data"]["token"], res["data"]["key"], res["data"]["url"], res["data"]["domain"])
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                with open(file_path, "rb") as f:
                    up = GLOBAL_SESSION.post(url=url, data={"token": token, "key": key}, files={"file": f}, timeout=60, proxies=proxies)
                up.raise_for_status()
                return f"{domain}/{key}"
            except requests.exceptions.RequestException:
                if attempt < max_attempts: time.sleep(1.0); continue
                raise
    except Exception as e: return f"Error: {str(e)}"
    return ""

class GrsaiAPI:
    def __init__(self, api_key: str, proxies: Optional[Dict] = None):
        if not api_key: raise Exception("API Key is empty")
        self.api_key = api_key
        self.session = GLOBAL_SESSION
        self.proxies = proxies
        self.auth_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _post_json(self, endpoint: str, data: Optional[Dict] = None, timeout: float = 300.0) -> Dict:
        url = f"https://grsai.dakka.com.cn{endpoint}"
        try:
            safe_timeout = max(5.0, float(timeout))
            resp = self.session.post(url, headers=self.auth_headers, json=data, timeout=safe_timeout, proxies=self.proxies)
            resp.raise_for_status()
        except Exception as e: raise Exception(f"Connection Error: {str(e)}")
        text = resp.text
        json_data = text[6:] if text.startswith("data: ") else text
        try: return json.loads(json_data)
        except: return {}

    def _poll_result(self, task_id: str, max_wait: float = 180.0) -> Dict:
        interval = 1.2
        start_poll = time.time()
        while True:
            if time.time() - start_poll > max_wait: raise Exception(f"Polling Timeout (User Limit Reached)")
            try: res = self._post_json("/v1/draw/result", {"id": task_id}, timeout=10)
            except: time.sleep(interval); continue
            status = str(res.get("status", "")).lower()
            if status in ("failed", "error"): raise Exception(f"Grsai FAIL: {res.get('error') or res}")
            if status in ("succeeded", "success", "done", "finished"): return res
            time.sleep(interval)

    def nano_banana_generate_image(self, prompt: str, model: str, urls: List[str], aspectRatio: str, imageSize: str, timeout: int = 180) -> Tuple[List, List, List]:
        start_t = time.time()
        payload = {"model": model, "prompt": prompt, "urls": urls, "shutProgress": True, "aspectRatio": aspectRatio, "imageSize": imageSize}
        request_timeout = max(30.0, float(timeout)) 
        print(f"[Grsai] Requesting {model} with size {imageSize}...") 
        first = self._post_json("/v1/draw/nano-banana", data=payload, timeout=request_timeout)
        status = str(first.get("status", "")).lower()
        if status in ("failed", "error"): raise Exception(f"Submission Fail: {first}")
        if status in ("succeeded", "success", "done", "finished") and first.get("results"): final_json = first
        else:
            task_id = first.get("id") or first.get("taskId")
            if not task_id: raise Exception(f"No Task ID: {first}")
            poll_time_left = timeout - (time.time() - start_t)
            if poll_time_left < 1: raise Exception("Budget exhausted")
            final_json = self._poll_result(task_id, max_wait=float(poll_time_left))
        result_urls = [r["url"] for r in final_json.get("results", []) if isinstance(r, dict) and r.get("url")]
        pils, errs, img_urls = [], [], []
        for u in result_urls:
            dl_timeout = max(timeout - (time.time() - start_t), 30.0)
            try:
                img = download_image_robust(u, timeout=dl_timeout, proxies=self.proxies)
                if is_black_512_image(img): errs.append(f"Black 512: {u}")
                else: pils.append(img); img_urls.append(u)
            except Exception as e:
                if self.proxies:
                    try:
                        img = download_image_robust(u, timeout=30.0, proxies=None)
                        if is_black_512_image(img): errs.append(f"Black 512: {u}")
                        else: pils.append(img); img_urls.append(u); continue
                    except: pass
                errs.append(f"DL Fail: {str(e)}")
        return pils, img_urls, errs

class EvolinkAPI:
    def __init__(self, api_key: str, proxies: Optional[Dict] = None):
        if not api_key: raise Exception("Evolink API Key is empty")
        self.api_key = api_key
        self.proxies = proxies
        self.base_url = "https://api.evolink.ai"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def upload_file(self, file_path: str) -> str:
        print(f"[Bridge] Uploading via Grsai Bridge: {file_path}")
        url = upload_file_zh(file_path, self.proxies, specific_key=self.api_key)
        if not url or url.startswith("Error"):
             fallback = get_grsai_api_key()
             if fallback and fallback != self.api_key:
                 print(f"[Bridge] Retrying with Fallback Grsai Key...")
                 url = upload_file_zh(file_path, self.proxies, specific_key=fallback)
        if not url or url.startswith("Error"): raise Exception(f"URL Gen Failed: {url}. Please ensure GRSAI_KEY is set.")
        print(f"[Bridge] URL Generated: {url}")
        return url

    def generate_image(self, prompt: str, model: str, urls: List[str], resolution: str, aspect_ratio: str, timeout: int = 180) -> Tuple[List, List, List]:
        print(f"[Debug] Evolink generate_image called with model: {model}")
        start_t = time.time()
        width, height = calculate_dimensions(resolution, aspect_ratio)
        actual_model = model
        if model == "seedream-4.5": actual_model = "doubao-seedream-4.5"
        payload = {"model": actual_model, "prompt": prompt, "num_images": 1, "safety_check": False}
        if actual_model == "doubao-seedream-4.5": payload["size"] = f"{width}x{height}"
        else: payload["width"] = width; payload["height"] = height
        if urls: payload["image_urls"] = urls
        headers_json = self.headers.copy()
        headers_json["Content-Type"] = "application/json"
        try:
            request_timeout = max(30.0, float(timeout))
            resp = GLOBAL_SESSION.post(f"{self.base_url}/v1/images/generations", headers=headers_json, json=payload, timeout=request_timeout, proxies=self.proxies)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise Exception(f"Evolink Submit Error: {str(e)}")
        task_id = data.get("id") or data.get("data", {}).get("id")
        if not task_id: raise Exception(f"No Task ID: {data}")
        result_url = None
        while time.time() - start_t < timeout:
            try:
                s_resp = GLOBAL_SESSION.get(f"{self.base_url}/v1/tasks/{task_id}", headers=self.headers, timeout=10, proxies=self.proxies)
                if s_resp.status_code == 200:
                    res_json = s_resp.json()
                    raw_status = res_json.get("status")
                    if raw_status is None and "data" in res_json: raw_status = res_json["data"].get("status")
                    status = str(raw_status).upper() if raw_status else ""
                    if status in ["FAILED", "ERROR", "CANCELLED", "TIMEOUT"]:
                        err = res_json.get("error") or "Unknown"
                        raise Exception(f"Task Status: {status} - {err}")
                    if status in ["COMPLETED", "SUCCEEDED", "SUCCESS", "DONE"]:
                        results = res_json.get("results") or res_json.get("output")
                        if not results and "data" in res_json: results = res_json["data"].get("results") or res_json["data"].get("output")
                        if results and len(results) > 0:
                            first = results[0]
                            if isinstance(first, str): result_url = first
                            elif isinstance(first, dict): result_url = first.get("url")
                            if result_url: break
            except Exception as e: print(f"[Evolink Poll Warn] {e}")
            time.sleep(1.5)
        if not result_url: raise Exception(f"Polling Timeout ({timeout}s) or No URL")
        dl_timeout = max(60, timeout - (time.time() - start_t))
        try:
            img = download_image_robust(result_url, timeout=dl_timeout, proxies=self.proxies)
            return [img], [result_url], []
        except Exception as e:
            if self.proxies:
                try:
                    img = download_image_robust(result_url, timeout=30.0, proxies=None)
                    return [img], [result_url], []
                except: pass
            raise Exception(f"Download Failed: {str(e)}")

# ==============================================================================
# 模块: Gemini Vision
# ==============================================================================

class ApiqikGeminiNode:
    def __init__(self): pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "platform": (["Grsai (grsaiapi.com)", "Evolink (evolink.ai)", "ChatAI (kaxsx.top)", "Jimiai (api.jimiai.ai)", "Apiqik (api.apiqik.com)", "Custom (使用下方自定义URL)"], {"default": "Grsai (grsaiapi.com)"}),
                "model": (["gemini-3-pro", "gemini-3.0-pro", "gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-pro-exp-0801", "gemini-pro-vision"], {"default": "gemini-3-pro"}),
            },
            "optional": {
                "trigger_value": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1, "display": "number"}),
                "delay_seconds": ("INT", {"default": 0, "min": 0, "max": 600, "step": 1, "display": "number"}),
                "stop_trigger": ("STRING", {"multiline": False, "default": "", "forceInput": False}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",), "image_5": ("IMAGE",),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "max_tokens": ("INT", {"default": 8192, "min": 128, "max": 1000000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "base_url": ("STRING", {"default": "https://grsaiapi.com/v1"}), 
                "ignore_ssl_verify": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("text",); FUNCTION = "generate_content"; CATEGORY = "Apiqik/Gemini"

    def tensor_to_base64_list(self, image_tensor):
        if image_tensor is None: return []
        results = []
        if not isinstance(image_tensor, torch.Tensor): return results
        for i in range(image_tensor.shape[0]):
            try:
                img_data = 255. * image_tensor[i].cpu().numpy()
                img = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                results.append(f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}")
            except Exception as e: print(f"⚠️ 图片转换失败: {e}"); continue
        return results

    def _robust_post_request(self, url, headers, json_data, ignore_ssl=True, max_retries=5):
        session = requests.Session()
        headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
        adapter = HTTPAdapter(max_retries=Retry(total=max_retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"]))
        session.mount("https://", adapter); session.mount("http://", adapter)
        def _do_request_stream(): 
            if "stream" in json_data: json_data["stream"] = True
            with session.post(url, headers=headers, json=json_data, timeout=(30, 600), verify=not ignore_ssl, stream=True) as response:
                if response.status_code != 200: return {"error_status": response.status_code, "text": response.text}
                collected_content = []
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: ") and decoded_line != "data: [DONE]":
                            try:
                                json_str = decoded_line[6:]
                                chunk = json.loads(json_str)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta: collected_content.append(delta["content"])
                            except: pass
                full_text = "".join(collected_content)
                if not full_text: return {"error": "Empty stream response"}
                return {"success_content": full_text}
        last_exception = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for attempt in range(max_retries):
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    future = executor.submit(_do_request_stream)
                    while not future.done():
                        if comfy.model_management.processing_interrupted(): future.cancel(); comfy.model_management.throw_exception_if_processing_interrupted()
                        time.sleep(0.1)
                    res = future.result()
                    if isinstance(res, dict):
                        if "error_status" in res:
                            if res["error_status"] == 503: time.sleep(3); continue 
                            raise ValueError(f"Status {res['error_status']}: {res['text']}")
                        if "success_content" in res: return res["success_content"]
                        if "error" in res: raise ValueError(res["error"])
                    return res
                except Exception as e:
                    if "Interrupt" in str(e): raise e
                    last_exception = e
                    print(f"⚠️ [Gemini] Retry {attempt+1}/{max_retries}: {str(e)}"); time.sleep(2)
        return {"error": str(last_exception)}

    def generate_content(self, api_key, prompt, platform, model, system_prompt="", stop_trigger="", delay_seconds=0, ignore_ssl_verify=True, trigger_value=1, 
                         image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, **kwargs):
        if trigger_value == 0: return ("0",)
        if len(str(stop_trigger).strip()) > 0: return ("Execution Skipped (Stop Trigger)",)
        if delay_seconds > 0:
            print(f"⏳ [Gemini] Waiting {delay_seconds}s...")
            for _ in range(delay_seconds): time.sleep(1); comfy.model_management.throw_exception_if_processing_interrupted()
        if not prompt or not prompt.strip(): return ("",)
        if not api_key: raise ValueError("❌ API Key Missing")
        image_inputs = [img for img in [image_1, image_2, image_3, image_4, image_5] if img is not None]
        url_map = {"Grsai": "https://grsaiapi.com/v1", "Evolink": "https://api.evolink.ai/v1", "ChatAI": "https://chatai.kaxsx.top/v1", "Jimiai": "https://api.jimiai.ai/v1", "Apiqik": "https://api.apiqik.com/v1"}
        base = next((url_map[k] for k in url_map if platform.startswith(k)), None)
        target_url = (base if base else kwargs.get('base_url', "https://grsaiapi.com/v1").strip()).rstrip('/')
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        content_list = [{"type": "text", "text": prompt}]
        for img_tensor in image_inputs:
            for b64 in self.tensor_to_base64_list(img_tensor):
                if b64: content_list.append({"type": "image_url", "image_url": {"url": b64, "detail": "low"}})
        messages.append({"role": "user", "content": content_list if len(content_list) > 1 else prompt})
        models_to_try = [model]
        if model != "gemini-2.5-pro": models_to_try.append("gemini-2.5-pro")
        if "gemini-1.5-pro" not in models_to_try: models_to_try.append("gemini-1.5-pro")
        last_error_msg = ""
        for i, current_model in enumerate(models_to_try):
            print(f"📡 [Gemini] Requesting: {current_model} ...")
            payload = {"model": current_model, "messages": messages, "temperature": kwargs.get("temperature"), "max_tokens": kwargs.get("max_tokens"), "stream": True}
            try:
                content_result = self._robust_post_request(f"{target_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json_data=payload, ignore_ssl=ignore_ssl_verify)
                if isinstance(content_result, str):
                    final_text = content_result
                    if "</think>" in final_text: final_text = final_text.split("</think>")[-1].strip()
                    print(f"✅ [Gemini] {current_model} 生成成功！"); return (final_text,)
                elif isinstance(content_result, dict) and "error" in content_result: raise ValueError(f"API Error: {content_result['error']}")
                else: raise ValueError(f"Unknown Response")
            except Exception as e:
                if "Interrupt" in str(e): raise e
                last_error_msg = str(e)
                print(f"⚠️ [Gemini] {current_model} Failed: {last_error_msg}")
                if i < len(models_to_try) - 1: time.sleep(1)
        raise ValueError(f"❌ [Gemini Failed] All models failed.\nLast Error: {last_error_msg}")

# ==============================================================================
# 模块: XZL Utility Suite (Updated Logic)
# ==============================================================================

class AspectRatioSelect:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "auto"], {"default": "16:9"}), "base_resolution": ("INT", {"default": 1024, "min": 512, "max": 8192, "step": 64})}, "optional": { "aspect_ratio_text": ("STRING", {"forceInput": True, "multiline": False}), }}
    RETURN_TYPES = (AnyType("*"), AnyType("*")); RETURN_NAMES = ("size_string", "ratio_cat"); FUNCTION = "get_formatted_size"; CATEGORY = "Utils/Resolution"
    def get_formatted_size(self, aspect_ratio, base_resolution, aspect_ratio_text=None):
        target_ratio_str = aspect_ratio_text.strip() if aspect_ratio_text and isinstance(aspect_ratio_text, str) and aspect_ratio_text.strip() else aspect_ratio
        if target_ratio_str.lower() == "auto" or target_ratio_str == "1:1": return (f"{base_resolution}x{base_resolution}", "1:1")
        try:
            parts = target_ratio_str.replace('x',':').replace('/',':').replace('：',':').split(':')
            if len(parts) < 2: return (f"{base_resolution}x{base_resolution}", "1:1")
            w_r, h_r = float(parts[0]), float(parts[1])
            ratio = w_r / h_r
            h = math.sqrt((base_resolution**2) / ratio); w = h * ratio
            return (f"{int(round(w/8)*8)}x{int(round(h/8)*8)}", target_ratio_str)
        except: return (f"{base_resolution}x{base_resolution}", "1:1")

class StoryboardCounter:
    CATEGORY = "xzl/utility"
    FUNCTION = "count"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "original_text")
    _global_count = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("STRING", {"default": "", "multiline": True, "forceInput": False, "placeholder": "在此输入提示词... 若包含'分镜X'则优先输出该词，否则自动计数"}),
                "prefix": ("STRING", {"default": "分镜", "multiline": False}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "Reset to 1 (重置为1)", "label_off": "Continue Counting (继续计数)"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": "Signal/Index (触发信号)"}),
            }
        }

    def count(self, trigger, prefix, reset, index):
        clean_trigger = ""
        if isinstance(trigger, list):
            clean_trigger = " ".join([str(s) for s in trigger])
        elif trigger is None:
            clean_trigger = ""
        else:
            clean_trigger = str(trigger)

        if reset:
            StoryboardCounter._global_count = 0

        pattern = re.escape(prefix) + r"\s*[零一二三四五六七八九十百]+"
        match = re.search(pattern, clean_trigger)

        if match:
            found_tag = match.group(0)
            found_tag_clean = found_tag.replace(" ", "").replace("\n", "")
            return (found_tag_clean, clean_trigger)
        else:
            StoryboardCounter._global_count += 1
            current_val = StoryboardCounter._global_count
            zh_num = int_to_zh(current_val)
            result_str = f"{prefix}{zh_num}"
            return (result_str, clean_trigger)

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("nan")

class RoleKeywordDetectorPro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": ""}),
                "target_keyword": (["图一角色", "图二角色", "图三角色", "图四角色", "图五角色", 
                                    "图六角色", "图七角色", "图八角色", "图九角色", "图十角色",
                                    "图十一角色", "图十二角色", "图十三角色", "图十四角色", "图十五角色"], {"default": "图一角色"}),
            },
        }
    RETURN_TYPES = ("INT", "STRING"); RETURN_NAMES = ("signal_int", "passthrough_text"); FUNCTION = "detect_role_in_large_text"; CATEGORY = "xzl/utility"
    def detect_role_in_large_text(self, text_input, target_keyword):
        return (1 if target_keyword in str(text_input or "") else 0, str(text_input or ""))

class SceneKeywordDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": ""}),
                "target_keyword": (["图一场景", "图二场景", "图三场景", "图四场景", "图五场景", 
                                    "图六场景", "图七场景", "图八场景", "图九场景", "图十场景"], {"default": "图一场景"}),
            },
        }
    RETURN_TYPES = ("INT", "STRING"); RETURN_NAMES = ("signal_int", "passthrough_text"); FUNCTION = "detect_scene_in_large_text"; CATEGORY = "xzl/utility"
    def detect_scene_in_large_text(self, text_input, target_keyword):
        return (1 if target_keyword in str(text_input or "") else 0, str(text_input or ""))

# ==============================================================================
# 修复的 Smart Image Concatenate 节点
# ==============================================================================

class SmartImageConcat:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (["right", "up"], {"default": "right", "label": "拼接方向"}),
                "match_image_size": ("BOOLEAN", {"default": True, "label": "强制匹配首图尺寸"}), 
                "output_mode": (["concatenate_all", "random_one"], {"default": "concatenate_all", "label": "输出模式"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": "随机种子"}),
            },
            "optional": {
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
                "image_4": ("IMAGE",), "image_5": ("IMAGE",), "image_6": ("IMAGE",),
                "image_7": ("IMAGE",), "image_8": ("IMAGE",), "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "concat_images"
    CATEGORY = "xzl/utility"

    def concat_images(self, direction, match_image_size, output_mode, seed, 
                      image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
                      image_6=None, image_7=None, image_8=None, image_9=None, image_10=None):
        
        all_inputs = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10]
        valid_images = []

        for img in all_inputs:
            if img is None: continue
            if img.shape[1] <= 64 or img.shape[2] <= 64: continue
            if img.max() < 0.001: continue
            valid_images.append(img)

        if not valid_images: return (torch.zeros((1, 64, 64, 3)),)

        if output_mode == "random_one":
            return (random.Random(seed).choice(valid_images),)

        processed_images = []
        if match_image_size:
            target_h, target_w = valid_images[0].shape[1], valid_images[0].shape[2]
            for img in valid_images:
                if img.shape[1] != target_h or img.shape[2] != target_w:
                    img_p = F.interpolate(img.permute(0, 3, 1, 2), size=(target_h, target_w), mode='bilinear', align_corners=False)
                    processed_images.append(img_p.permute(0, 2, 3, 1))
                else:
                    processed_images.append(img)
        else:
            processed_images = valid_images

        try:
            return (torch.cat(processed_images, dim=2 if direction == "right" else 1),)
        except RuntimeError:
            return (valid_images[0],)

class BatchIntegerGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9999999, "step": 1, "label": "索引(Page/Row)"}),
                "batch_size": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1, "label": "每行数量(Count)"}),
                "start_from": ("INT", {"default": 1, "min": 0, "step": 1, "label": "起始数字(通常为1)"}),
            },
        }
    RETURN_TYPES = ("INT", "STRING"); RETURN_NAMES = ("int_list", "debug_string"); OUTPUT_IS_LIST = (True, False); FUNCTION = "generate_sequence"; CATEGORY = "xzl/utility"
    def generate_sequence(self, index, batch_size, start_from):
        ints = [start_from + (index * batch_size) + i for i in range(batch_size)]
        return (ints, f"Index {index}: {ints}")

class _GrsaiNodeBase:
    FUNCTION = "execute"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")
    def _create_error_image(self): return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    def _handle_image_uploads_generic(self, images_in: List, provider: str, client: Any, proxies: Optional[Dict]) -> Tuple[Union[List, Dict], List]:
        urls, temps = [], []
        if not any(img is not None for img in images_in): return urls, temps
        try:
            for i, t in enumerate(images_in):
                if t is None: continue
                pil = tensor_to_pil(t)[0]
                rgb = safe_pil_to_rgb(pil)
                with tempfile.NamedTemporaryFile(suffix=f"_{i}.png", delete=False) as tmp:
                    rgb.save(tmp, "PNG"); temps.append(tmp.name)
            if not temps: return [], []
            for p in temps:
                if provider == "grsai":
                    up_res = upload_file_zh(p, proxies, specific_key=client.api_key)
                    if up_res.startswith("Error"): return {"error": up_res}, temps
                    urls.append(up_res)
                elif provider == "evolink":
                    if hasattr(client, 'upload_file'): urls.append(client.upload_file(p))
                    else: return {"error": "Client upload not supported"}, temps
            return urls, temps
        except Exception as e: return {"error": str(e)}, temps
    def _cleanup_temp_files(self, files):
        for p in files:
            try: os.unlink(p)
            except: pass

class GrsaiProviderSelector:
    CATEGORY = "Nkxx/Utilities"
    @classmethod
    def INPUT_TYPES(s): return {"required": {"mode": (["grsai", "evolink", "local_sequence"], {"default": "grsai"})}}
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("provider_val",); FUNCTION = "get_mode"
    def get_mode(self, mode): return (mode,)

class GrsaiNanoBanana(_GrsaiNodeBase):
    CATEGORY = "Nkxx/图像"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["grsai", "evolink", "local_sequence"], {"default": "grsai"}),
                "grsai_model": (SUPPORTED_MODELS, {"default": "nano-banana-pro"}),
                "value": ("INT", {"default": 1, "min": 0, "max": 1}),
                "prompt": ("STRING", {"multiline": True, "default": "Cat"}),
                "resolution": (SUPPORTED_RESOLUTIONS, {"default": "1K"}),
                "concurrency": ("INT", {"default": 1}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "timeout_seconds": ("INT", {"default": 60, "min": 5}),
                "retry_count": ("INT", {"default": 2}),
                "trigger_interval": ("INT", {"default": 0}),
                "trigger_index": ("INT", {"default": 0}),
            },
            "optional": {
                "trigger_offset": ("INT", {"default": 0, "min": 0}),
                "fallback_image": ("IMAGE",), "provider_override": ("STRING", {"forceInput": True}),
                "directory_path": ("STRING", {"default": ""}), "sequence_index": ("INT", {"default": 0}),
                "sort_method": (["natural", "alphabetical"], {"default": "natural"}),
                "api_key": ("STRING", {"default": ""}), "proxy_url": ("STRING", {"default": ""}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",), "image_5": ("IMAGE",),
                "video_1": ("VIDEO",), "character_id": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "VIDEO", "STRING")
    RETURN_NAMES = ("image", "status", "failed", "video_conn", "character_id")

    def execute(self, provider, grsai_model, value, prompt, resolution, concurrency, aspect_ratio, timeout_seconds, retry_count=2,
                trigger_interval=0, trigger_index=0, api_key="", proxy_url="", directory_path="", sequence_index=0, 
                sort_method="natural", provider_override=None, trigger_offset=0, 
                image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, video_1=None, character_id="", 
                **kwargs):
        video_pass, char_pass = video_1, character_id
        def wrap(img, msg, code): return {"ui": {"string": [msg]}, "result": (img, msg, code, video_pass, char_pass)}

        if value == 0:
            raw_fallback = kwargs.get("fallback_image")
            if raw_fallback is not None: return wrap(raw_fallback, "Bypassed (Original Input)", 0)
            return wrap(self._create_error_image(), "Bypassed (No Input Connected)", 0)

        if trigger_interval > 0:
            if (trigger_index - trigger_offset) % trigger_interval != 0: return wrap(self._create_error_image(), f"Trigger Skipped ({trigger_index})", 0)

        fail_safe_image = kwargs.get("fallback_image") if kwargs.get("fallback_image") is not None else self._create_error_image()
        active_provider = provider_override.strip().lower() if (provider_override and provider_override.strip()) else provider.strip().lower()

        if active_provider == "local_sequence":
            if not directory_path or not os.path.isdir(directory_path): return wrap(fail_safe_image, "Invalid Path", 1)
            try:
                files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png','.jpg','.webp'))]
                if not files: return wrap(fail_safe_image, "No Images", 1)
                files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)] if sort_method == "natural" else None)
                idx = sequence_index % len(files)
                img = Image.open(os.path.join(directory_path, files[idx])).convert("RGB")
                return wrap(torch.from_numpy(np.array(img).astype(np.float32)/255.0).unsqueeze(0), f"Local: {files[idx]}", 0)
            except Exception as e: return wrap(fail_safe_image, str(e), 1)

        final_key = api_key.strip()
        if not final_key:
            if active_provider == "grsai": final_key = get_grsai_api_key()
            elif active_provider == "evolink": final_key = os.getenv("EVOLINK_KEY", "").strip() or get_grsai_api_key()
        if not final_key: return wrap(fail_safe_image, f"No API Key: {active_provider}", 1)

        proxies = format_proxies(proxy_url)
        client = GrsaiAPI(final_key, proxies) if active_provider == "grsai" else EvolinkAPI(final_key, proxies)
        
        imgs_in = [image_1, image_2, image_3, image_4, image_5]
        urls, temps = [], []
        if any(img is not None for img in imgs_in):
            upload_res, temps = self._handle_image_uploads_generic(imgs_in, active_provider, client, proxies)
            if isinstance(upload_res, dict) and "error" in upload_res:
                self._cleanup_temp_files(temps); return wrap(fail_safe_image, f"Upload Fail: {upload_res['error']}", 1)
            urls = upload_res

        deadline, last_err = time.time() + timeout_seconds, "Unknown"
        try:
            for attempt in range(retry_count + 1):
                budget = deadline - time.time()
                if budget <= 1: last_err = "Timeout"; break
                try:
                    pils, _, errs = [], [], []
                    if active_provider == "grsai": pils, _, errs = client.nano_banana_generate_image(prompt, grsai_model, urls, aspect_ratio, resolution, budget)
                    elif active_provider == "evolink": pils, _, errs = client.generate_image(prompt, grsai_model, urls, resolution, aspect_ratio, budget)
                    if pils: return wrap(pil_to_tensor(pils), f"[{active_provider.upper()}] Success", 0)
                    last_err = str(errs)
                except Exception as e: last_err = str(e)
                if attempt < retry_count: time.sleep(2)
            
            # Removed fallback logic to gemini-3-pro-image-preview here as requested

            return wrap(fail_safe_image, f"All Failed: {last_err}", 1)
        except Exception as e:
            traceback.print_exc()
            return wrap(fail_safe_image, f"Crash: {str(e)}", 1)
        finally: self._cleanup_temp_files(temps)

class NkxxSafeImageFromBatch:
    CATEGORY = "Nkxx/Image"
    @classmethod
    def INPUT_TYPES(s): return {"required": {"start": ("INT", {"default": 0}), "length": ("INT", {"default": 1}), "trigger_refresh": ("INT", {"default": 0})}, "optional": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("image",); FUNCTION = "execute"
    @classmethod
    def IS_CHANGED(s, **kwargs): return float("NaN")
    def execute(self, start, length, trigger_refresh, image=None):
        ph = lambda: torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        try:
            if image is None or image.numel() == 0 or start >= image.shape[0]: return (ph(),)
            res = image[start : start + length]
            return (res,) if res.shape[0] > 0 else (ph(),)
        except: return (ph(),)

# ==============================================================================
# 5. 节点注册映射
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "ApiqikGeminiNode": ApiqikGeminiNode,
    "AspectRatioSelect": AspectRatioSelect,
    "StoryboardCounter": StoryboardCounter,
    "RoleKeywordDetectorPro": RoleKeywordDetectorPro,
    "SceneKeywordDetector": SceneKeywordDetector, 
    "SmartImageConcat": SmartImageConcat,
    "BatchIntegerGenerator": BatchIntegerGenerator,
    "GrsaiProviderSelector": GrsaiProviderSelector,
    "GrsaiNanoBanana": GrsaiNanoBanana,
    "NkxxSafeImageFromBatch": NkxxSafeImageFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApiqikGeminiNode": "Apiqik/Gemini Vision (Clean Output)",
    "AspectRatioSelect": "Aspect Ratio",
    "StoryboardCounter": "Storyboard Counter (分镜计数器)",
    "RoleKeywordDetectorPro": "Role Keyword Detector (角色关键词识别)",
    "SceneKeywordDetector": "Scene Keyword Detector (场景关键词识别)",
    "SmartImageConcat": "Smart Image Concatenate (智能图像拼接)",
    "BatchIntegerGenerator": "Batch Integer Gen (按行整数生成器)",
    "GrsaiProviderSelector": "🍌 Grsai Provider Selector",
    "GrsaiNanoBanana": "🍌 Grsai Nano Banana (Pro/Fast + Local)",
    "NkxxSafeImageFromBatch": "🔧 Safe Image From Batch (Empty Allowed)",
}
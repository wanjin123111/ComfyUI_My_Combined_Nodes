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

print("Loading Custom Nodes: Grsai & XZL Utility Suite (Standalone __init__)...")

# ==============================================================================
# 全局工具与配置
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

class SaveTextFile:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        default_remove = (
            "same outfit\n"
            "同样的场景\n"
            "same color tone\n"
            "相同色调\n"
            "character: same as previons\n"
            "character: same as previous\n"
            "与前一个相同的场景\n"
            "必须严格参考图像\n"
            "strict reference"
        )
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "my_text", "label": "文件名"}),
                "save_mode": (["create_new", "overwrite", "append"], {"default": "create_new", "label": "保存模式"}),
                "min_length": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "label": "最小字数限制"}),
                "remove_terms": ("STRING", {"default": default_remove, "multiline": True, "label": "剔除词(保存前删除)"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "placeholder": "自定义路径 (留空存至output)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "xzl/utility"

    def save_text(self, text, filename_prefix, save_mode, min_length, remove_terms, custom_path=""):
        content_to_save = ""
        if isinstance(text, list):
            valid_texts = [str(t) for t in text if t is not None]
            content_to_save = "\n".join(valid_texts)
        else:
            content_to_save = str(text)

        terms_to_remove = [t.strip() for t in remove_terms.split('\n') if t.strip()]
        for term in terms_to_remove:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            content_to_save = pattern.sub("", content_to_save)

        lines = content_to_save.splitlines()
        clean_lines = [line.strip() for line in lines if line.strip()]
        content_to_save = "\n".join(clean_lines)

        if len(content_to_save) < min_length: return ("",)

        full_output_folder = custom_path.strip() if custom_path.strip() else self.output_dir
        if not os.path.exists(full_output_folder):
            try: os.makedirs(full_output_folder, exist_ok=True)
            except: return ("",)

        valid_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in (' ', '_', '-')).strip() or "text_output"
        file_path = ""
        mode = 'w' 
        if save_mode == "create_new":
            counter = 1
            while True:
                file_name = f"{valid_prefix}_{counter:05}.txt"
                file_path = os.path.join(full_output_folder, file_name)
                if not os.path.exists(file_path): break
                counter += 1
        elif save_mode == "append":
            mode = 'a'
            file_path = os.path.join(full_output_folder, f"{valid_prefix}.txt")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                content_to_save = "\n" + content_to_save
        else: 
            file_path = os.path.join(full_output_folder, f"{valid_prefix}.txt")

        try:
            with open(file_path, mode, encoding="utf-8") as f: f.write(content_to_save)
        except Exception as e: print(f"[SaveText] Error: {e}")
        return (file_path,)

class CustomPathSave:
    def __init__(self): self.run_index = 0
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "output_path": ("STRING", {"default": r"C:\ComfyUI_Output", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "Image", "label": "默认前缀(无输入时使用)"}),
                "extension": (["png", "jpg", "webp"], ),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "filename_input": ("*", {"forceInput": True, "label": "文件名输入(连整数列表)"}), 
                "manual_index": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "label": "强制索引(Manual Index)"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("INT",); RETURN_NAMES = ("run_index",); FUNCTION = "save_images"; OUTPUT_NODE = True; CATEGORY = "xzl/utility"; INPUT_IS_LIST = True

    def save_images(self, images, output_path, filename_prefix, extension, quality, filename_input=None, manual_index=None, prompt=None, extra_pnginfo=None):
        actual_manual_index = -1
        if manual_index is not None:
            if isinstance(manual_index, list):
                 if len(manual_index) > 0: actual_manual_index = int(manual_index[0])
            else: actual_manual_index = int(manual_index)

        if actual_manual_index > -1:
            self.run_index = actual_manual_index
            current_seq_id = actual_manual_index
        else:
            self.run_index += 1
            current_seq_id = self.run_index

        img_batch = images[0] 
        path = output_path[0]
        prefix = filename_prefix[0]
        ext = extension[0]
        qual = quality[0]
        p = prompt[0] if prompt and len(prompt) > 0 else None
        ep = extra_pnginfo[0] if extra_pnginfo and len(extra_pnginfo) > 0 else None

        if not os.path.exists(path):
            try: os.makedirs(path, exist_ok=True)
            except: pass

        results = list()
        custom_names = []
        use_custom_names = False

        if filename_input is not None:
            raw_input = filename_input
            target_list = []
            if isinstance(raw_input, list):
                if len(raw_input) > 0 and isinstance(raw_input[0], list): target_list = raw_input[0]
                else: target_list = raw_input
            if len(target_list) > 0:
                custom_names = [str(x) for x in target_list]
                use_custom_names = True

        for idx, image in enumerate(img_batch):
            if torch.isnan(image).any() or image.max() < 0.001: continue

            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if ext == 'png':
                metadata = PngInfo()
                if p is not None: metadata.add_text("prompt", json.dumps(p))
                if ep is not None:
                    for x in ep: metadata.add_text(x, json.dumps(ep[x]))

            file_name = ""
            if use_custom_names:
                if idx < len(custom_names): file_name = f"{custom_names[idx]}.{ext}"
                else: file_name = f"{prefix}_{idx}.{ext}"
            else:
                counter = 1
                while True:
                    file_name = f"{prefix}_{counter}.{ext}"
                    if not os.path.exists(os.path.join(path, file_name)): break
                    counter += 1

            full_path = os.path.join(path, file_name)
            try:
                if ext == 'png': img.save(full_path, pnginfo=metadata, compress_level=4)
                elif ext == 'webp': img.save(full_path, quality=qual, lossless=False)
                else: img.save(full_path, quality=qual, optimize=True)
                results.append({"filename": file_name, "subfolder": path, "type": "output"})
            except Exception as e: print(f"[CustomPathSave] Error saving {file_name}: {e}")

        return { "ui": { "images": results }, "result": (current_seq_id,) }

class LoadNewestTextFile:
    _read_history = set()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False, "placeholder": "C:\\ComfyUI_Output\\Text"}),
                "manual_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": "触发器 (Seed)"}),
            },
            "optional": {
                "trigger": ("STRING", {"forceInput": True}),
                "reset_history": ("BOOLEAN", {"default": False, "label": "强制重置记忆"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT"); RETURN_NAMES = ("text", "filename", "status_code (1=Old/Read)"); FUNCTION = "load_newest"; CATEGORY = "xzl/utility"

    def load_newest(self, folder_path, manual_index, trigger=None, reset_history=False):
        if reset_history: LoadNewestTextFile._read_history.clear()
        if not folder_path or not os.path.exists(folder_path): return ("", "", 1)

        try:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(".txt")]
        except: return ("", "", 1)
        if not files: return ("", "", 1)

        try:
            newest_file = max(files, key=os.path.getmtime)
            newest_file_abs = os.path.abspath(newest_file)
            if time.time() - os.path.getmtime(newest_file_abs) > 300: return ("", "", 1)
        except: return ("", "", 1)

        if newest_file_abs in LoadNewestTextFile._read_history: return ("", "", 1)

        try:
            with open(newest_file, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
            LoadNewestTextFile._read_history.add(newest_file_abs)
            return (content, os.path.basename(newest_file), 0)
        except: return ("", "", 1)

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

class SceneKeywordMapper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": ""}),
                "default_index": ("INT", {"default": 0, "min": 0, "max": 99}),
            },
        }
    RETURN_TYPES = ("INT", "STRING"); RETURN_NAMES = ("scene_index", "passthrough_text"); FUNCTION = "map_scene"; CATEGORY = "xzl/utility"
    def map_scene(self, text_input, default_index):
        c, r = str(text_input or ""), default_index
        for i, k in enumerate(["图一场景", "图二场景", "图三场景", "图四场景", "图五场景", "图六场景", "图七场景", "图八场景", "图九场景", "图十场景"]):
            if k in c: r = i; break
        return (r, c)

# ==============================================================================
# 修复的 Smart Image Concatenate 节点 (已更新为10图+过滤版)
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
        
        all_inputs = [image_1, image_2, image_3, image_4, image_5, 
                      image_6, image_7, image_8, image_9, image_10]
        valid_images = []

        for img in all_inputs:
            if img is None: continue
            
            h = img.shape[1]
            w = img.shape[2]
            
            # [过滤] 1. 过滤尺寸过小的图像 (<= 64 像素)
            if h <= 64 or w <= 64:
                continue
                
            # [过滤] 2. 过滤全黑图像 (亮度阈值 < 0.001)
            if img.max() < 0.001:
                continue

            valid_images.append(img)

        # 如果没有有效图像，返回一个小的黑色块防止报错
        if not valid_images:
            return (torch.zeros((1, 64, 64, 3)),)

        # 模式：随机选择一张
        if output_mode == "random_one":
            rng = random.Random(seed)
            selected_img = rng.choice(valid_images)
            return (selected_img,)

        # 模式：全部拼接
        processed_images = []

        if match_image_size:
            target_h = valid_images[0].shape[1]
            target_w = valid_images[0].shape[2]
            for img in valid_images:
                h = img.shape[1]
                w = img.shape[2]
                if h != target_h or w != target_w:
                    # 调整尺寸以匹配第一张图
                    img_p = img.permute(0, 3, 1, 2)
                    img_p = F.interpolate(img_p, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    img = img_p.permute(0, 2, 3, 1)
                processed_images.append(img)
        else:
            processed_images = valid_images

        # 确定拼接维度 (2=宽度/右, 1=高度/上)
        concat_dim = 2 if direction == "right" else 1
        
        try:
            final_image = torch.cat(processed_images, dim=concat_dim)
        except RuntimeError as e:
            print(f"[SmartImageConcat] Error: {e}")
            return (valid_images[0],)

        return (final_image,)

# ==============================================================================
# 新增节点: 智能过滤拼接 (5输入/严格模式)
# ==============================================================================

class SmartFilterConcatFive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (["right", "up"], {"default": "right", "label": "拼接方向"}),
                "match_image_size": ("BOOLEAN", {"default": True, "label": "强制匹配首图尺寸"}),
            },
            "optional": {
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
                "image_4": ("IMAGE",), "image_5": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "xzl/utility"

    def execute(self, direction, match_image_size, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        # 1. 收集输入
        inputs = [image_1, image_2, image_3, image_4, image_5]
        valid_images = []

        # 2. 严格过滤逻辑
        for img in inputs:
            # 判空
            if img is None: continue
            # 判黑 (阈值 0.001)
            if img.max() < 0.001: continue
            # 判无效尺寸 (可选，防止极小图报错)
            if img.shape[1] < 4 or img.shape[2] < 4: continue
            
            valid_images.append(img)

        # 3. 如果没有有效图 -> 输出黑块
        if not valid_images:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        # 4. 如果只有一个有效图 -> 直接输出
        if len(valid_images) == 1:
            return (valid_images[0],)

        # 5. 多个图 -> 拼接
        processed_images = []
        if match_image_size:
            # 以第一张图为基准
            target_h = valid_images[0].shape[1]
            target_w = valid_images[0].shape[2]
            for img in valid_images:
                if img.shape[1] != target_h or img.shape[2] != target_w:
                    # 调整尺寸: Permute (B,H,W,C) -> (B,C,H,W) -> Interpolate -> Permute back
                    img_p = img.permute(0, 3, 1, 2)
                    img_p = F.interpolate(img_p, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    img = img_p.permute(0, 2, 3, 1)
                processed_images.append(img)
        else:
            processed_images = valid_images

        # 拼接方向: 2=宽(right), 1=高(up)
        dim = 2 if direction == "right" else 1
        
        # 处理 Batch Size 不一致的情况 (防御性编程)
        try:
            max_b = max(img.shape[0] for img in processed_images)
            final_stack = []
            for img in processed_images:
                if img.shape[0] < max_b:
                    img = img.repeat(max_b, 1, 1, 1)
                final_stack.append(img)
            
            result = torch.cat(final_stack, dim=dim)
        except Exception as e:
            print(f"[SmartFilterConcatFive] Concat Error: {e}")
            return (valid_images[0],) # 回退到第一张

        return (result,)

class LoadImagesFromPathSequential:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "multiline": False, "placeholder": "X:/path/to/your/images"}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "sort_method": (["natural", "alphabetical"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT"); RETURN_NAMES = ("image", "mask", "filename_text", "current_index"); FUNCTION = "load_images"; CATEGORY = "xzl/utility"

    def load_images(self, directory_path, start_index, batch_size, sort_method):
        if not os.path.isdir(directory_path): raise FileNotFoundError(f"Directory not found: {directory_path}")
        files = [f for f in os.listdir(directory_path) if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}]
        if sort_method == "natural": files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])
        else: files.sort()
        if not files: raise ValueError("No valid images found in directory.")

        images_list, masks_list, filenames = [], [], []
        for i in range(batch_size):
            file_name = files[(start_index + i) % len(files)]
            img = ImageOps.exif_transpose(Image.open(os.path.join(directory_path, file_name)))
            mask = 1. - torch.from_numpy(np.array(img.getchannel('A')).astype(np.float32)/255.) if 'A' in img.getbands() else torch.zeros((img.height, img.width), dtype=torch.float32)
            images_list.append(torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32)/255.))
            masks_list.append(mask)
            filenames.append(file_name)

        if batch_size > 1: return (torch.stack(images_list), torch.stack(masks_list), ", ".join(filenames), start_index)
        return (images_list[0].unsqueeze(0), masks_list[0].unsqueeze(0), filenames[0], start_index)

class LoadImageByIndexSmart:
    def __init__(self): self.cache_key, self.cache_result = None, None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:\\path\\to\\images", "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "label": "索引 (Index)"}),
                "trigger_interval": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1, "label": "触发间隔 (Interval)"}),
                "refresh_cache": ("BOOLEAN", {"default": False, "label_on": "Always Read Disk (强制刷盘)", "label_off": "Use Cache (使用缓存)"}),
            },
        }
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT"); RETURN_NAMES = ("image", "filename", "full_path", "caption_text", "image_count"); FUNCTION = "load_image_smart"; CATEGORY = "xzl/utility"
    @classmethod
    def IS_CHANGED(cls, folder_path, index, trigger_interval, refresh_cache): return float("nan") if refresh_cache else float("nan")

    def load_image_smart(self, folder_path, index, trigger_interval, refresh_cache):
        folder_path = folder_path.strip().strip('"').strip("'")
        effective_index = index - (index % trigger_interval) if trigger_interval > 1 else index
        current_key = (folder_path, effective_index)
        if not refresh_cache and self.cache_result is not None and self.cache_key == current_key: return self.cache_result

        if effective_index == 0:
            row1 = [[0.0,0.0,0.0], [0.0,1.0,0.0], [0.85,0.25,0.05]]
            row2 = [[0.85,0.25,0.05], [0.3,0.0,0.0], [0.0,0.0,0.0]]
            data = F.interpolate(torch.tensor([row1, row2], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2), size=(512, 768), mode='nearest').permute(0, 2, 3, 1)
            res = (data, "Special_Pattern_00.png", "Internal_Code", "Start/Cover Image", 1)
            self.cache_key, self.cache_result = current_key, res; return res

        if not os.path.isdir(folder_path): return (torch.zeros((1, 64, 64, 3)), "", "", "", 0)
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'))], key=lambda x: int(re.match(r'^(\d+)', x).group(1)) if re.match(r'^(\d+)', x) else float('inf'))
        if not files: return (torch.zeros((1, 64, 64, 3)), "", "", "", 0)

        target_filename = files[(effective_index - 1) % len(files)]
        try:
            img = ImageOps.exif_transpose(Image.open(os.path.join(folder_path, target_filename)).convert('RGB'))
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
            txt_p = os.path.join(folder_path, os.path.splitext(target_filename)[0] + ".txt")
            cap = open(txt_p, 'r', encoding='utf-8', errors='ignore').read() if os.path.exists(txt_p) else ""
            res = (img_tensor, target_filename, os.path.join(folder_path, target_filename), cap, len(files))
            self.cache_key, self.cache_result = current_key, res; return res
        except: return (torch.zeros((1, 64, 64, 3)), target_filename, "", "", len(files))

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

class ParallelImageHub:
    _storage = [None] * 10
    _empty_image = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"reset": ("BOOLEAN", {"default": False})},
            "optional": {
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
                "image_4": ("IMAGE",), "image_5": ("IMAGE",), "image_6": ("IMAGE",),
                "image_7": ("IMAGE",), "image_8": ("IMAGE",), "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
            }
        }
    RETURN_TYPES = tuple(["IMAGE"] * 10); RETURN_NAMES = tuple([f"Output {i}" for i in range(1, 11)]); FUNCTION = "route"; CATEGORY = "Router"
    @classmethod
    def IS_CHANGED(s, **kwargs): return float("nan")
    def route(self, reset, 
              image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
              image_6=None, image_7=None, image_8=None, image_9=None, image_10=None):
        if ParallelImageHub._empty_image is None: ParallelImageHub._empty_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        if reset: ParallelImageHub._storage = [None] * 10
        inputs = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10]
        for i in range(10):
            if inputs[i] is not None: ParallelImageHub._storage[i] = inputs[i]
            if ParallelImageHub._storage[i] is None: ParallelImageHub._storage[i] = ParallelImageHub._empty_image
        return tuple(ParallelImageHub._storage)

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
            if active_provider == "evolink" and grsai_model == "nano-banana-2-lite":
                try:
                    pils, _, _ = client.generate_image(prompt, "gemini-3-pro-image-preview", urls, resolution, aspect_ratio, float(timeout_seconds))
                    if pils: return wrap(pil_to_tensor(pils), "[EVOLINK] Success (Auto-Downgrade)", 0)
                except: pass
            return wrap(fail_safe_image, f"All Failed: {last_err}", 1)
        except Exception as e:
            traceback.print_exc()
            return wrap(fail_safe_image, f"Crash: {str(e)}", 1)
        finally: self._cleanup_temp_files(temps)

class GrsaiNanoBananaBatch(GrsaiNanoBanana):
    CATEGORY = "Nkxx/图像"
    @classmethod
    def INPUT_TYPES(cls):
        base = GrsaiNanoBanana.INPUT_TYPES()
        req = {
            "provider": (["grsai", "evolink"], {"default": "grsai"}),
            "grsai_model": base["required"]["grsai_model"],
            "file_path": ("STRING", {"default": "", "placeholder": "CSV/Excel Path"}),
            "column_name": ("STRING", {"default": "prompt"}),
            "prompt_prefix": ("STRING", {"multiline": True}),
            "concurrency": base["required"]["concurrency"],
            "max_count": ("INT", {"default": 50}),
            "resolution": base["required"]["resolution"],
            "aspect_ratio": base["required"]["aspect_ratio"],
            "timeout_seconds": base["required"]["timeout_seconds"],
            "retry_count": base["required"]["retry_count"],
        }
        return {"required": req, "optional": base["optional"]}
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images_batch", "status", "failed")

    def execute(self, provider, grsai_model, file_path, column_name, prompt_prefix, concurrency, max_count, resolution, aspect_ratio, timeout_seconds, retry_count=2, provider_override=None, 
                image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, video_1=None, character_id="",
                **kwargs):
        if pd is None: return {"ui": {"string": ["Pandas Missing"]}, "result": (self._create_error_image(), "Pandas Lib Missing", 1)}
        active_provider = provider_override.strip().lower() if (provider_override and provider_override.strip()) else provider.strip().lower()
        if active_provider == "local_sequence": return {"ui": {"string": ["No Local"]}, "result": (self._create_error_image(), "No Local", 1)}
        
        final_key = kwargs.get("api_key", "").strip()
        if not final_key:
             if active_provider == "grsai": final_key = get_grsai_api_key()
             elif active_provider == "evolink": final_key = os.getenv("EVOLINK_KEY", "")
        if not final_key: return {"ui": {"string": ["No Key"]}, "result": (self._create_error_image(), "No Key", 1)}
        
        try:
            if file_path.endswith('.csv'): df = pd.read_csv(file_path)
            else: df = pd.read_excel(file_path)
            prompts = [f"{prompt_prefix}{p}" for p in df[column_name].dropna().astype(str).tolist()[:max_count]]
        except: return {"ui": {"string": ["File/Column Error"]}, "result": (self._create_error_image(), "File/Column Error", 1)}
        
        proxies = format_proxies(kwargs.get("proxy_url",""))
        client = GrsaiAPI(final_key, proxies) if active_provider == "grsai" else EvolinkAPI(final_key, proxies)
        
        imgs_in = [image_1, image_2, image_3, image_4, image_5]
        urls, temps = [], []
        if any(img is not None for img in imgs_in):
             upload_res, temps = self._handle_image_uploads_generic(imgs_in, active_provider, client, proxies)
             if isinstance(upload_res, dict) and "error" in upload_res:
                 self._cleanup_temp_files(temps); return {"ui": {"string": ["Upload Fail"]}, "result": (self._create_error_image(), upload_res["error"], 1)}
             urls = upload_res
        
        all_imgs, global_errs = [], []
        for i, p in enumerate(prompts):
            item_end = time.time() + timeout_seconds
            success = False
            for attempt in range(retry_count + 1):
                if item_end - time.time() < 1: break
                try:
                    pils, _, _ = client.nano_banana_generate_image(p, grsai_model, urls, aspect_ratio, resolution, item_end - time.time()) if active_provider == "grsai" else client.generate_image(p, grsai_model, urls, resolution, aspect_ratio, item_end - time.time())
                    if pils: all_imgs.extend(pils); success = True; break
                except: pass
                time.sleep(1)
            if not success: global_errs.append(f"Item {i+1} Failed")
        self._cleanup_temp_files(temps)
        if not all_imgs: return {"ui": {"string": ["Batch Failed"]}, "result": (self._create_error_image(), f"Failures: {len(global_errs)}", 1)}
        return {"ui": {"string": ["Batch Done"]}, "result": (pil_to_tensor(all_imgs), f"Done ({len(all_imgs)})", 0)}

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

class GrsaiLLMWriter(_GrsaiNodeBase):
    CATEGORY = "Nkxx/语言模型"
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model": (["gemini-2.5-flash"],), "main_prompt": ("STRING", {}), "system_prompt": ("STRING", {}), "output_filename": ("STRING", {}), "column_name": ("STRING", {})}}
    RETURN_TYPES = ("STRING", "STRING"); RETURN_NAMES = ("file_path", "status"); 
    def execute(self, **kwargs): return {"ui": {"string": ["Placeholder"]}, "result": ("", "Placeholder")}

# ==============================================================================
# 5. 节点注册映射
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "ApiqikGeminiNode": ApiqikGeminiNode,
    "AspectRatioSelect": AspectRatioSelect,
    "StoryboardCounter": StoryboardCounter,
    "SaveTextFile": SaveTextFile,
    "CustomPathSave": CustomPathSave,
    "LoadNewestTextFile": LoadNewestTextFile,
    "RoleKeywordDetectorPro": RoleKeywordDetectorPro,
    "SceneKeywordDetector": SceneKeywordDetector, 
    "SceneKeywordMapper": SceneKeywordMapper,
    "SmartImageConcat": SmartImageConcat,
    "SmartFilterConcatFive": SmartFilterConcatFive,
    "LoadImagesFromPathSequential": LoadImagesFromPathSequential,
    "LoadImageByIndexSmart": LoadImageByIndexSmart,
    "BatchIntegerGenerator": BatchIntegerGenerator,
    "ParallelImageHub": ParallelImageHub,
    "GrsaiProviderSelector": GrsaiProviderSelector,
    "GrsaiNanoBanana": GrsaiNanoBanana,
    "GrsaiNanoBananaBatch": GrsaiNanoBananaBatch,
    "NkxxSafeImageFromBatch": NkxxSafeImageFromBatch,
    "GrsaiLLMWriter": GrsaiLLMWriter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApiqikGeminiNode": "Apiqik/Gemini Vision (Clean Output)",
    "AspectRatioSelect": "Aspect Ratio",
    "StoryboardCounter": "Storyboard Counter (分镜计数器)",
    "SaveTextFile": "Save Text File (保存文本文件)",
    "CustomPathSave": "Save Image Custom (自定义路径保存)",
    "LoadNewestTextFile": "Load Newest Text (读取最新文本-5min时效)",
    "RoleKeywordDetectorPro": "Role Keyword Detector (角色关键词识别)",
    "SceneKeywordDetector": "Scene Keyword Detector (场景关键词识别)",
    "SceneKeywordMapper": "Scene Keyword Mapper (场景关键词映射)",
    "SmartImageConcat": "Smart Image Concatenate (智能图像拼接)",
    "SmartFilterConcatFive": "Smart Filter & Concat (5-Input Strict)",
    "LoadImagesFromPathSequential": "Sequence Image Loader (Batch & Natural)",
    "LoadImageByIndexSmart": "Load Image Sequence (Index/Numeric)",
    "BatchIntegerGenerator": "Batch Integer Gen (按行整数生成器)",
    "ParallelImageHub": "Parallel Image Hub (10-Way)",
    "GrsaiProviderSelector": "🍌 Grsai Provider Selector",
    "GrsaiNanoBanana": "🍌 Grsai Nano Banana (Pro/Fast + Local)",
    "GrsaiNanoBananaBatch": "🍌 Grsai Nano Banana Batch",
    "NkxxSafeImageFromBatch": "🔧 Safe Image From Batch (Empty Allowed)",
    "GrsaiLLMWriter": "✍️ Grsai LLM/VLM Writer",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
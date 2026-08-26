#!/usr/bin/env python3
"""
BLIP服务器客户端库
用于从本地调用服务器上的BLIP模型

使用示例:
    from blip_client import BlipClient
    
    # 初始化客户端（只需设置一次）
    client = BlipClient("http://your-server-ip:5000")
    
    # 视觉问答
    result = client.visual_qa("image.jpg", "what is in the image?")
    print(result['answer'])
    
    # 图像描述
    result = client.image_caption("image.jpg")
    print(result['caption'])
"""

import requests
import base64
from PIL import Image
import io


class BlipClient:
    """BLIP服务器客户端"""
    
    def __init__(self, server_url="http://localhost:5000", timeout=60, connect_timeout=5):
        """
        初始化客户端
        
        参数:
            server_url: 服务器地址，例如 "http://192.168.1.100:5000"
            timeout: 请求超时时间（秒），默认60秒
            connect_timeout: 连接超时时间（秒），默认5秒
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.connect_timeout = connect_timeout
    
    def _image_to_base64(self, image_path_or_pil):
        """将图片转换为base64字符串"""
        if isinstance(image_path_or_pil, str):
            # 文件路径
            with open(image_path_or_pil, 'rb') as f:
                image_bytes = f.read()
                return base64.b64encode(image_bytes).decode('utf-8')
        else:
            # PIL Image对象
            buffer = io.BytesIO()
            image_path_or_pil.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
    
    def check_health(self):
        """
        检查服务器健康状态
        
        返回:
            dict: 服务器状态信息
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=self.connect_timeout
            )
            return response.json()
        except requests.exceptions.Timeout:
            return {"status": "unhealthy", "error": f"连接超时（{self.connect_timeout}秒）"}
        except requests.exceptions.ConnectionError:
            return {"status": "unhealthy", "error": "无法连接到服务器"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def visual_qa(self, image_path_or_pil, question, max_new_tokens=50):
        """
        视觉问答
        
        参数:
            image_path_or_pil: 图片路径（str）或PIL Image对象
            question: 问题（str）
            max_new_tokens: 最大生成token数（默认50）
        
        返回:
            dict: {
                "success": bool,
                "question": str,
                "answer": str,
                "processing_time": float
            }
        """
        image_base64 = self._image_to_base64(image_path_or_pil)
        
        try:
            response = requests.post(
                f"{self.server_url}/visual_qa",
                json={
                    "image": image_base64,
                    "question": question,
                    "max_new_tokens": max_new_tokens
                },
                timeout=(self.connect_timeout, self.timeout)  # (连接超时, 读取超时)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"请求超时（连接:{self.connect_timeout}秒，读取:{self.timeout}秒）"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": f"无法连接到服务器 {self.server_url}"}
        except requests.exceptions.HTTPError as e:
            return {"success": False, "error": f"HTTP错误: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def image_caption(self, image_path_or_pil, max_new_tokens=50):
        """
        生成图像描述
        
        参数:
            image_path_or_pil: 图片路径（str）或PIL Image对象
            max_new_tokens: 最大生成token数（默认50）
        
        返回:
            dict: {
                "success": bool,
                "caption": str,
                "processing_time": float
            }
        """
        image_base64 = self._image_to_base64(image_path_or_pil)
        
        try:
            response = requests.post(
                f"{self.server_url}/image_caption",
                json={
                    "image": image_base64,
                    "max_new_tokens": max_new_tokens
                },
                timeout=(self.connect_timeout, self.timeout)  # (连接超时, 读取超时)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"请求超时（连接:{self.connect_timeout}秒，读取:{self.timeout}秒）"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": f"无法连接到服务器 {self.server_url}"}
        except requests.exceptions.HTTPError as e:
            return {"success": False, "error": f"HTTP错误: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def batch_visual_qa(self, image_paths_or_pils, question, max_new_tokens=50):
        """
        批量视觉问答（多张图片同一个问题）
        
        参数:
            image_paths_or_pils: 图片路径列表或PIL Image对象列表
            question: 问题（str）
            max_new_tokens: 最大生成token数（默认50）
        
        返回:
            dict: {
                "success": bool,
                "question": str,
                "results": [
                    {"index": int, "success": bool, "answer": str, "processing_time": float},
                    ...
                ],
                "total_images": int,
                "total_time": float,
                "avg_time": float
            }
        """
        images_base64 = [self._image_to_base64(img) for img in image_paths_or_pils]
        
        response = requests.post(
            f"{self.server_url}/batch_visual_qa",
            json={
                "images": images_base64,
                "question": question,
                "max_new_tokens": max_new_tokens
            },
            timeout=self.timeout * len(image_paths_or_pils)  # 批量处理需要更长时间
        )
        
        return response.json()


# 为了向后兼容，也可以提供模块级函数
# 但需要先初始化一个全局客户端
_global_client = None

def init(server_url="http://localhost:5000", timeout=60):
    """
    初始化全局客户端（可选，如果不想使用类的话）
    
    参数:
        server_url: 服务器地址
        timeout: 请求超时时间（秒）
    """
    global _global_client
    _global_client = BlipClient(server_url, timeout)

def visual_qa(image_path_or_pil, question, max_new_tokens=50):
    """视觉问答（模块级函数）"""
    if _global_client is None:
        raise RuntimeError("请先调用 blip_client.init(server_url) 初始化")
    return _global_client.visual_qa(image_path_or_pil, question, max_new_tokens)

def image_caption(image_path_or_pil, max_new_tokens=50):
    """图像描述（模块级函数）"""
    if _global_client is None:
        raise RuntimeError("请先调用 blip_client.init(server_url) 初始化")
    return _global_client.image_caption(image_path_or_pil, max_new_tokens)

def batch_visual_qa(image_paths_or_pils, question, max_new_tokens=50):
    """批量视觉问答（模块级函数）"""
    if _global_client is None:
        raise RuntimeError("请先调用 blip_client.init(server_url) 初始化")
    return _global_client.batch_visual_qa(image_paths_or_pils, question, max_new_tokens)


#!/home/ycs/miniconda3/envs/blip/bin/python

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import time
import os

# 设置 cuDNN
torch.backends.cudnn.benchmark = True

app = Flask(__name__)
CORS(app)  # 允许跨域请求

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = "/home/ycs/blip_server/blip2-opt-2.7b"

# 全局变量存储模型
model = None
processor = None

def print_memory_usage(stage=""):
    """打印显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[显存] {stage}: 已分配={allocated:.2f}GB, 已缓存={reserved:.2f}GB")

def load_model():
    """加载BLIP模型"""
    global model, processor
    
    if model is not None:
        return  # 模型已加载
    
    print("正在加载 processor...")
    try:
        processor = Blip2Processor.from_pretrained(model_path, use_fast=True)
    except:
        processor = Blip2Processor.from_pretrained(model_path, use_fast=False)
    
    print("正在加载模型...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True
    )
    print("模型加载完成！")
    print_memory_usage("模型加载后")

def decode_image(image_data):
    """解码图片数据，支持base64字符串或文件路径"""
    if isinstance(image_data, str):
        # 如果是base64编码的字符串
        if image_data.startswith('data:image'):
            # 处理 data:image/png;base64,xxx 格式
            image_data = image_data.split(',')[1]
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return image
        except:
            # 如果不是base64，尝试作为文件路径
            if os.path.exists(image_data):
                return Image.open(image_data).convert('RGB')
            else:
                raise ValueError(f"无法解析图片数据: {image_data[:50]}...")
    else:
        raise ValueError("图片数据格式不支持")

def visual_qa(image, question, max_new_tokens=50):
    """对图像进行视觉问答"""
    prompt = f"Question: {question} Answer:"
    
    # 处理输入
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    # 确保所有输入张量都在正确的设备上，浮点张量转换为 float16
    inputs = {
        k: v.to(device).to(torch.float16) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point 
        else v.to(device) if isinstance(v, torch.Tensor) 
        else v 
        for k, v in inputs.items()
    }
    
    # 生成答案
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # 解码答案
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 清理显存
    del inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generated_text

def image_caption(image, max_new_tokens=50):
    """生成图像描述"""
    inputs = processor(images=image, return_tensors="pt")
    # 确保所有输入张量都在正确的设备上，浮点张量转换为 float16
    inputs = {
        k: v.to(device).to(torch.float16) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point 
        else v.to(device) if isinstance(v, torch.Tensor) 
        else v 
        for k, v in inputs.items()
    }
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 清理显存
    del inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generated_text

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    })

@app.route('/visual_qa', methods=['POST'])
def api_visual_qa():
    """视觉问答API接口"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求体为空"}), 400
        
        # 获取图片数据
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "缺少图片数据"}), 400
        
        # 获取问题
        question = data.get('question', '')
        if not question:
            return jsonify({"error": "缺少问题"}), 400
        
        # 获取可选参数
        max_new_tokens = data.get('max_new_tokens', 50)
        
        # 解码图片
        start_time = time.time()
        image = decode_image(image_data)
        
        # 执行视觉问答
        answer = visual_qa(image, question, max_new_tokens)
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": answer,
            "processing_time": round(processing_time, 3)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/image_caption', methods=['POST'])
def api_image_caption():
    """图像描述API接口"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求体为空"}), 400
        
        # 获取图片数据
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "缺少图片数据"}), 400
        
        # 获取可选参数
        max_new_tokens = data.get('max_new_tokens', 50)
        
        # 解码图片
        start_time = time.time()
        image = decode_image(image_data)
        
        # 生成图像描述
        caption = image_caption(image, max_new_tokens)
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "caption": caption,
            "processing_time": round(processing_time, 3)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/batch_visual_qa', methods=['POST'])
def api_batch_visual_qa():
    """批量视觉问答API接口（多张图片同一个问题）"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求体为空"}), 400
        
        # 获取图片数据列表
        images_data = data.get('images')
        if not images_data or not isinstance(images_data, list):
            return jsonify({"error": "缺少图片数据列表"}), 400
        
        # 获取问题
        question = data.get('question', '')
        if not question:
            return jsonify({"error": "缺少问题"}), 400
        
        # 获取可选参数
        max_new_tokens = data.get('max_new_tokens', 50)
        
        # 处理所有图片
        results = []
        total_start_time = time.time()
        
        for i, image_data in enumerate(images_data):
            try:
                image = decode_image(image_data)
                start_time = time.time()
                answer = visual_qa(image, question, max_new_tokens)
                processing_time = time.time() - start_time
                
                results.append({
                    "index": i,
                    "success": True,
                    "answer": answer,
                    "processing_time": round(processing_time, 3)
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        total_time = time.time() - total_start_time
        
        return jsonify({
            "success": True,
            "question": question,
            "results": results,
            "total_images": len(images_data),
            "total_time": round(total_time, 3),
            "avg_time": round(total_time / len(images_data), 3)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # 启动时加载模型
    load_model()
    
    # 启动服务器
    print("=" * 60)
    print("BLIP2 服务器启动中...")
    print(f"设备: {device}")
    print("=" * 60)
    print("\n可用接口:")
    print("  GET  /health - 健康检查")
    print("  POST /visual_qa - 视觉问答")
    print("  POST /image_caption - 图像描述")
    print("  POST /batch_visual_qa - 批量视觉问答")
    print("\n服务器启动在 http://0.0.0.0:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, threaded=True)


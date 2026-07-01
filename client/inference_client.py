import os
import time
import csv
import numpy as np
import cv2
from PIL import Image
import tritonclient.http as httpclient

try:
    with open("client/imagenet_classes.txt", "r") as f:
        IMAGENET_LABELS = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Warning: client/imagenet_classes.txt not found. Using generic labels.")
    IMAGENET_LABELS = [f"Class_{i}" for i in range(1000)]
except Exception as e:
    print(f"Error loading labels: {e}")
    IMAGENET_LABELS = [f"Class_{i}" for i in range(1000)]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
    "hairdryer", "toothbrush"
]

def log_result(model_name, image_name, label, confidence, latency):
    os.makedirs("logs", exist_ok=True)
    file_path = "logs/inference_history.csv"
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model Name", "Image", "Prediction", "Confidence", "Latency"])
        writer.writerow([model_name, image_name, label, f"{confidence:.4f}", f"{latency:.4f}"])

class TritonInferenceClient:
    def __init__(self, url="localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)

    def get_model_params(self, model_name):
        meta = self.client.get_model_metadata(model_name)
        input_info = meta['inputs'][0]
        output_info = meta['outputs'][0]
        
        # Get H and W from the end of the shape list (ignoring batch dim)
        full_shape = input_info['shape']
        h, w = full_shape[-2], full_shape[-1]
        
        return {
            "input_name": input_info['name'],
            "output_name": output_info['name'],
            "h": h, "w": w
        }

    def preprocess(self, image_path, h, w, is_resnet):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((w, h))
        img_data = np.array(img).astype(np.float32) / 255.0

        if is_resnet:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_data = (img_data - mean) / std

        img_data = np.transpose(img_data, (2, 0, 1))
        return np.expand_dims(img_data, axis=0).astype(np.float32)

    def postprocess_resnet(self, output_data):
        # Apply Softmax to get probabilities 0..1
        scores = np.squeeze(output_data)
        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()
        
        idx = np.argmax(probs)
        label = IMAGENET_LABELS[idx] if idx < len(IMAGENET_LABELS) else f"ID_{idx}"
        return [{"label": label, "score": float(probs[idx])}]

    def postprocess_yolo(self, output_data, conf_thresh=0.35):
        # Формат YOLOv8/11: [1, 84, 8400] -> [8400, 84]
        predictions = np.squeeze(output_data)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
            
        class_probs = predictions[:, 4:]
        scores = np.max(class_probs, axis=1)
        
        mask = scores > conf_thresh
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(predictions) == 0: return []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        boxes[:, 0] -= boxes[:, 2] / 2 # x_min
        boxes[:, 1] -= boxes[:, 3] / 2 # y_min
        
        # NMS removes overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, 0.45)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "label": COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else f"ID_{class_ids[i]}",
                    "score": float(scores[i])
                })
        # Сортируем по уверенности
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def run(self, model_name, image_path, conf_thresh=0.35):
        is_resnet = "resnet" in model_name.lower()
        params = self.get_model_params(model_name)
        input_data = self.preprocess(image_path, params['h'], params['w'], is_resnet)
        
        inputs = [httpclient.InferInput(params['input_name'], input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        start_time = time.time()
        try:
            res = self.client.infer(model_name=model_name, inputs=inputs)
        except Exception as e:
            print(f"Triton inference failed for {model_name}: {e}")
            raise
        latency = time.time() - start_time
        raw_output = res.as_numpy(params['output_name'])
        
        return (self.postprocess_resnet(raw_output) if is_resnet else self.postprocess_yolo(raw_output, conf_thresh)), is_resnet, latency

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    triton_url = os.getenv("TRITON_URL", "localhost:8000")
    triton = TritonInferenceClient(url=triton_url)
    models = ["resnet50_onnx", "resnet50_notonnx", "yolo_onnx", "yolo_notonnx"]
    image_dir = "client/samples"
    
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} not found!")
        exit()

    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for model_name in models:
        print(f"\n{'='*50}\n📊 MODEL: {model_name}\n{'='*50}")
        
        for img_name in images:
            try:
                results, is_resnet, latency = triton.run(model_name, os.path.join(image_dir, img_name))
                
                if is_resnet:
                    # For ResNet, we just print the top classification result
                    res = results[0]
                    print(f"🖼️  [{img_name}] -> CLASSIFICATION: {res['label']} {res['score']*100:.2f}% | Latency: {latency*1000:.1f}ms")
                    log_result(model_name, img_name, res['label'], res['score'], latency)
                else:
                    # For YOLO, we print the list of all detected objects
                    print(f"🖼️  [{img_name}] -> DETECTED {len(results)} objects | Latency: {latency*1000:.1f}ms")
                    if not results:
                        print("     (nothing found)")
                        log_result(model_name, img_name, "None", 0.0, latency)
                    for i, r in enumerate(results):
                        print(f"     - {r['label']:<15} | confidence: {r['score']*100:>6.2f}%")
                        if i == 0:  # Log only the most confident detection
                            log_result(model_name, img_name, r['label'], r['score'], latency)
                        
            except Exception as e:
                print(f"❌ Error on {img_name}: {e}")

    # ==================================================
    # ⏱️ BENCHMARKING MODE
    # ==================================================
    if len(images) > 0:
        bench_img = os.path.join(image_dir, images[0])
        print(f"\n{'='*60}\n🚀 BENCHMARKING: 50 Iterations on {images[0]}\n{'='*60}")
        print(f"{'Model Name':<20} | {'Avg Latency (ms)':<15} | {'Throughput (req/s)':<18}")
        print("-" * 60)
        
        bench_results = {}
        for model_name in models:
            latencies = []
            
            # WARMUP
            for _ in range(3):
                try: triton.run(model_name, bench_img)
                except Exception as e:
                    print(f"Warmup failed for {model_name}: {e}")
                    break
                
            # BENCHMARK
            start_total = time.time()
            for _ in range(50):
                try:
                    _, _, lat = triton.run(model_name, bench_img)
                    latencies.append(lat)
                except Exception as e:
                    print(f"Benchmark iteration failed for {model_name}: {e}")
                    break
            
            if len(latencies) == 50:
                total_time = time.time() - start_total
                avg_lat = np.mean(latencies) * 1000
                tpu = 50 / total_time
                print(f"{model_name:<20} | {avg_lat:<15.2f} | {tpu:<18.2f}")
                bench_results[model_name] = {"lat": avg_lat, "tps": tpu}
            else:
                print(f"{model_name:<20} | FAILED")
                
        # --- COMPARISON ---
        print(f"\n{'='*60}")
        print("🏆 ONNX vs PyTorch Comparison")
        print(f"{'='*60}")
        
        if "resnet50_onnx" in bench_results and "resnet50_notonnx" in bench_results:
            o_tps = bench_results["resnet50_onnx"]["tps"]
            pt_tps = bench_results["resnet50_notonnx"]["tps"]
            diff = ((o_tps - pt_tps) / pt_tps) * 100 if pt_tps > 0 else 0
            print(f"ResNet50: ONNX has {'+' if diff > 0 else ''}{diff:.1f}% throughput vs PyTorch ({o_tps:.2f} vs {pt_tps:.2f} req/s)")
            
        if "yolo_onnx" in bench_results and "yolo_notonnx" in bench_results:
            o_tps = bench_results["yolo_onnx"]["tps"]
            pt_tps = bench_results["yolo_notonnx"]["tps"]
            diff = ((o_tps - pt_tps) / pt_tps) * 100 if pt_tps > 0 else 0
            print(f"YOLO:     ONNX has {'+' if diff > 0 else ''}{diff:.1f}% throughput vs PyTorch ({o_tps:.2f} vs {pt_tps:.2f} req/s)")
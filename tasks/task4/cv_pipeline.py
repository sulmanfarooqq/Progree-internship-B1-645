"""
TASK 4: Mini Project — Computer Vision Object Detector & Frame-By-Frame Segmenter
Built with Pillow (PIL) — adaptive thresholding, Gaussian filtering, contour extraction.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import time
import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# ─── 1. VIDEO FRAME GENERATOR ───────────────────────────────────────────────

class SyntheticVideoStream:
    """Generates synthetic video frames with moving shapes for CV pipeline testing."""

    SHAPE_TYPES = ['circle', 'rectangle', 'triangle', 'star']

    def __init__(self, width=640, height=480, n_frames=30):
        self.width = width
        self.height = height
        self.n_frames = n_frames
        self.shapes = self._init_shapes()
        self.frame_count = 0

    def _init_shapes(self):
        shapes = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255)]
        for i in range(6):
            shapes.append({
                'type': self.SHAPE_TYPES[i % len(self.SHAPE_TYPES)],
                'color': colors[i],
                'pos': [
                    np.random.randint(50, self.width - 100),
                    np.random.randint(50, self.height - 100)
                ],
                'size': np.random.randint(20, 50),
                'velocity': [
                    np.random.choice([-3, -2, 2, 3]),
                    np.random.choice([-3, -2, 2, 3])
                ],
            })
        return shapes

    def generate_frame(self):
        img = Image.new('RGB', (self.width, self.height), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        gray_bg = Image.new('L', (self.width, self.height), color=30)
        draw_gray = ImageDraw.Draw(gray_bg)
        gray_val = 180

        for shape in self.shapes:
            x, y = shape['pos']
            s = shape['size']
            color = shape['color']
            gray_c = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])

            if shape['type'] == 'circle':
                draw.ellipse([x - s, y - s, x + s, y + s], fill=color)
                draw_gray.ellipse([x - s, y - s, x + s, y + s], fill=gray_c)
            elif shape['type'] == 'rectangle':
                draw.rectangle([x - s, y - s // 2, x + s, y + s // 2], fill=color)
                draw_gray.rectangle([x - s, y - s // 2, x + s, y + s // 2], fill=gray_c)
            elif shape['type'] == 'triangle':
                draw.polygon([(x, y - s), (x - s, y + s), (x + s, y + s)], fill=color)
                draw_gray.polygon([(x, y - s), (x - s, y + s), (x + s, y + s)], fill=gray_c)
            elif shape['type'] == 'star':
                points = []
                for i in range(10):
                    angle = i * math.pi / 5 - math.pi / 2
                    r = s if i % 2 == 0 else s // 2
                    points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
                draw.polygon(points, fill=color)
                draw_gray.polygon(points, fill=gray_c)

        # Update positions
        for shape in self.shapes:
            shape['pos'][0] += shape['velocity'][0]
            shape['pos'][1] += shape['velocity'][1]
            if shape['pos'][0] < s or shape['pos'][0] > self.width - s:
                shape['velocity'][0] *= -1
            if shape['pos'][1] < s or shape['pos'][1] > self.height - s:
                shape['velocity'][1] *= -1

        self.frame_count += 1
        return np.array(img), np.array(gray_bg)


# ─── 2. IMAGE PROCESSING FUNCTIONS ─────────────────────────────────────────

def apply_gaussian_filter(image, kernel_size=5):
    """Apply Gaussian blur using PIL."""
    img = Image.fromarray(image)
    if len(image.shape) == 3:
        img = img.convert('L')
    return np.array(img.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2)))


def adaptive_threshold(image, block_size=11, c=2):
    """Fast adaptive thresholding using integral image."""
    h, w = image.shape[:2]
    img_float = image.astype(np.float32)
    integral = np.cumsum(np.cumsum(img_float, axis=0), axis=1)
    half = block_size // 2
    result = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        y1 = max(0, y - half)
        y2 = min(h - 1, y + half)
        for x in range(w):
            x1 = max(0, x - half)
            x2 = min(w - 1, x + half)
            area = (y2 - y1 + 1) * (x2 - x1 + 1)
            sum_val = integral[y2, x2]
            if y1 > 0:
                sum_val -= integral[y1 - 1, x2]
            if x1 > 0:
                sum_val -= integral[y2, x1 - 1]
            if y1 > 0 and x1 > 0:
                sum_val += integral[y1 - 1, x1 - 1]
            threshold = sum_val / area - c
            result[y, x] = 255 if img_float[y, x] > threshold else 0
    return result


def detect_contours(binary_image, min_area=50):
    """Simple contour detection by finding connected components."""
    h, w = binary_image.shape
    visited = np.zeros((h, w), dtype=bool)
    contours = []

    for y in range(h):
        for x in range(w):
            if binary_image[y, x] > 0 and not visited[y, x]:
                # BFS to find connected component
                component_pixels = []
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    component_pixels.append((cy, cx))
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and binary_image[ny, nx] > 0 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                if len(component_pixels) >= min_area:
                    ys = [p[0] for p in component_pixels]
                    xs = [p[1] for p in component_pixels]
                    contours.append({
                        'pixels': component_pixels,
                        'bbox': (min(xs), min(ys), max(xs), max(ys)),
                        'area': len(component_pixels),
                        'centroid': (int(np.mean(xs)), int(np.mean(ys))),
                    })
    return contours


def classify_shape(contour):
    """Classify shape based on contour bounding box aspect ratio and area ratio."""
    x1, y1, x2, y2 = contour['bbox']
    w = x2 - x1
    h = y2 - y1
    aspect = w / h if h > 0 else 1
    bbox_area = w * h
    fill_ratio = contour['area'] / bbox_area if bbox_area > 0 else 0

    if abs(aspect - 1.0) < 0.3 and fill_ratio > 0.7:
        return 'circle'
    elif abs(aspect - 1.0) < 0.3 and fill_ratio < 0.6:
        return 'triangle'
    elif aspect > 1.5 or aspect < 0.67:
        return 'rectangle'
    elif fill_ratio > 0.5 and fill_ratio < 0.8:
        return 'star'
    else:
        return 'unknown'


# ─── 3. FRAME-BY-FRAME PIPELINE ─────────────────────────────────────────────

class ObjectDetectionPipeline:
    def __init__(self):
        self.tracking_history = {}
        self.frame_metrics = []
        self.next_id = 0

    def process_frame(self, frame_gray, frame_idx):
        start_time = time.perf_counter()

        # Step 1: Gaussian filtering
        blurred = apply_gaussian_filter(frame_gray, kernel_size=5)

        # Step 2: Adaptive thresholding
        binary = adaptive_threshold(blurred, block_size=15, c=3)

        # Step 3: Morphological cleanup (using threshold adjustment instead of dilation)
        h, w = binary.shape
        binary = binary[::2, ::2]
        binary = np.kron(binary, np.ones((2, 2), dtype=np.uint8))[:h, :w]

        # Step 4: Contour detection
        contours = detect_contours(binary, min_area=50)

        # Step 5: Shape classification
        detections = []
        for c in contours:
            shape_type = classify_shape(c)
            detections.append({
                'shape': shape_type,
                'bbox': c['bbox'],
                'centroid': c['centroid'],
                'area': c['area'],
            })
            self._track_object(c['centroid'], shape_type, frame_idx)

        elapsed = time.perf_counter() - start_time

        metrics = {
            'frame': frame_idx,
            'processing_time_ms': elapsed * 1000,
            'n_contours': len(contours),
            'n_detections': len(detections),
            'detection_rate': len(detections) / max(len(contours), 1),
            'total_area': sum(c['area'] for c in contours) if contours else 0,
        }
        self.frame_metrics.append(metrics)
        return binary, detections, metrics

    def _dilate(self, image, iterations=1):
        result = image.copy()
        h, w = image.shape
        for _ in range(iterations):
            dilated = result.copy()
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if np.any(result[y - 1:y + 2, x - 1:x + 2] > 0):
                        dilated[y, x] = 255
            result = dilated
        return result

    def _track_object(self, centroid, shape_type, frame_idx):
        obj_id = None
        for oid, history in self.tracking_history.items():
            last_pos = history['positions'][-1]
            dist = np.sqrt((centroid[0] - last_pos[0]) ** 2 + (centroid[1] - last_pos[1]) ** 2)
            if dist < 50 and history['type'] == shape_type:
                obj_id = oid
                break
        if obj_id is None:
            obj_id = self.next_id
            self.next_id += 1
            self.tracking_history[obj_id] = {
                'type': shape_type,
                'positions': [],
                'frames_seen': 0,
            }
        self.tracking_history[obj_id]['positions'].append(centroid)
        self.tracking_history[obj_id]['frames_seen'] += 1


# ─── 4. VISUALIZATION ────────────────────────────────────────────────────────

def create_detection_visualization(frame_idx, frame_gray, binary, detections, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Original
    axes[0].imshow(frame_gray, cmap='gray')
    axes[0].set_title(f'Frame {frame_idx} — Grayscale Input', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Binary
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f'Frame {frame_idx} — Thresholded + Contours', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        color_map = {'circle': '#3498DB', 'rectangle': '#E74C3C', 'triangle': '#2ECC71', 'star': '#F39C12', 'unknown': '#95A5A6'}
        color = color_map.get(d['shape'], '#FFF')
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2)
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 5, d['shape'], color=color, fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.7, pad=2))

    # Detection overlay
    overlay = np.stack([frame_gray] * 3, axis=-1).astype(np.float32)
    overlay = overlay / 255.0
    for d in detections:
        cx, cy = d['centroid']
        color_map = {'circle': [0.2, 0.6, 0.9], 'rectangle': [0.9, 0.3, 0.2],
                     'triangle': [0.2, 0.8, 0.4], 'star': [0.95, 0.6, 0.1], 'unknown': [0.6, 0.6, 0.6]}
        color = color_map.get(d['shape'], [1, 1, 1])
        cv_sz = int(np.sqrt(d['area'] / math.pi))
        # Draw centroid marker
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if 0 <= cy + dy < overlay.shape[0] and 0 <= cx + dx < overlay.shape[1]:
                    overlay[cy + dy, cx + dx] = color
        # Bounding box
        x1, y1, x2, y2 = d['bbox']
        overlay[y1:y1 + 2, x1:x2] = color
        overlay[y2 - 2:y2, x1:x2] = color
        overlay[y1:y2, x1:x1 + 2] = color
        overlay[y1:y2, x2 - 2:x2] = color

    axes[2].imshow(overlay)
    axes[2].set_title(f'Frame {frame_idx} — Detections Overlay', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pipeline_metrics(metrics, save_path):
    frames = [m['frame'] for m in metrics]
    times = [m['processing_time_ms'] for m in metrics]
    detections = [m['n_detections'] for m in metrics]
    areas = [m['total_area'] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(frames, times, color='#3498DB', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Per-Frame Processing Time', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(np.mean(times), color='red', linestyle='--', label=f'Avg: {np.mean(times):.2f}ms')
    axes[0, 0].legend()

    axes[0, 1].plot(frames, detections, color='#2ECC71', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Detections Per Frame', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Objects Detected')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(np.mean(detections), color='red', linestyle='--', label=f'Avg: {np.mean(detections):.1f}')
    axes[0, 1].legend()

    axes[1, 0].plot(frames, areas, color='#E74C3C', linewidth=2, marker='^', markersize=4)
    axes[1, 0].set_title('Total Contour Area Per Frame', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Pixel Area')
    axes[1, 0].grid(True, alpha=0.3)

    det_rates = [m['detection_rate'] for m in metrics]
    axes[1, 1].plot(frames, det_rates, color='#F39C12', linewidth=2, marker='d', markersize=4)
    axes[1, 1].set_title('Detection Rate Per Frame', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Detection Rate')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_frame_timeline_grid(frames_data, save_path):
    """Show a grid of sampled frames across the video timeline."""
    n_samples = min(8, len(frames_data))
    indices = np.linspace(0, len(frames_data) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        frame, binary, detections = frames_data[idx]
        axes[i].imshow(frame, cmap='gray')
        axes[i].set_title(f'Frame {idx}', fontsize=11, fontweight='bold')
        axes[i].axis('off')
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            color_map = {'circle': '#3498DB', 'rectangle': '#E74C3C', 'triangle': '#2ECC71', 'star': '#F39C12', 'unknown': '#95A5A6'}
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color_map.get(d['shape'], '#FFF'), linewidth=2)
            axes[i].add_patch(rect)
    plt.suptitle('Frame Timeline — Object Detection Pipeline', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_tracking_visualization(tracking_history, save_path):
    """Plot tracking paths of all detected objects."""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

    for i, (obj_id, history) in enumerate(tracking_history.items()):
        positions = history['positions']
        if len(positions) > 1:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            color = colors[i % len(colors)]
            ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=4, label=f"ID {obj_id}: {history['type']}")
            ax.scatter([xs[0]], [ys[0]], color=color, s=100, marker='o', zorder=5)
            ax.scatter([xs[-1]], [ys[-1]], color=color, s=120, marker='s', zorder=5)

    ax.set_title('Object Tracking History (Frame-by-Frame)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── 5. MAIN PIPELINE ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TASK 4: Computer Vision - Object Detector and Segmenter Pipeline")
    print("=" * 60)

    print("\n[1/5] Initializing video stream...")
    stream = SyntheticVideoStream(width=640, height=480, n_frames=30)
    pipeline = ObjectDetectionPipeline()
    print(f"   Resolution: 640x480 | Frames: 30 | Shapes: {len(stream.shapes)}")

    print("\n[2/5] Processing frames through pipeline...")
    frames_data = []
    for i in range(stream.n_frames):
        frame_rgb, frame_gray = stream.generate_frame()
        binary, detections, metrics = pipeline.process_frame(frame_gray, i)
        frames_data.append((frame_gray, binary, detections))

        if (i + 1) % 5 == 0:
            print(f"   Frame {i + 1}/{stream.n_frames} - {metrics['n_detections']} detections in {metrics['processing_time_ms']:.2f}ms")

    print("\n[3/5] Generating visualizations...")

    # Sample frame visualizations
    for i in [0, 5, 10, 15, 20, 25]:
        if i < len(frames_data):
            save_path = OUTPUT_DIR / f'frame_{i:04d}_detection.png'
            create_detection_visualization(i, frames_data[i][0], frames_data[i][1], frames_data[i][2], save_path)
    print(f"   Saved 6 frame detection visualizations")

    # Metrics plot
    metrics_path = OUTPUT_DIR / 'pipeline_metrics.png'
    plot_pipeline_metrics(pipeline.frame_metrics, metrics_path)
    print(f"   Saved: pipeline_metrics.png")

    # Timeline grid
    timeline_path = OUTPUT_DIR / 'frame_timeline.png'
    plot_frame_timeline_grid(frames_data, timeline_path)
    print(f"   Saved: frame_timeline.png")

    # Tracking viz
    tracking_path = OUTPUT_DIR / 'tracking_paths.png'
    plot_tracking_visualization(pipeline.tracking_history, tracking_path)
    print(f"   Saved: tracking_paths.png")

    print("\n[4/5] Computing summary statistics...")
    all_metrics = pipeline.frame_metrics
    avg_time = np.mean([m['processing_time_ms'] for m in all_metrics])
    avg_detections = np.mean([m['n_detections'] for m in all_metrics])
    total_frames = len(all_metrics)
    total_objects = len(pipeline.tracking_history)

    summary = {
        'total_frames': total_frames,
        'resolution': '640x480',
        'objects_tracked': total_objects,
        'avg_processing_time_ms': round(avg_time, 3),
        'avg_detections_per_frame': round(avg_detections, 2),
        'total_detections': int(sum(m['n_detections'] for m in all_metrics)),
        'pipeline_stages': ['Gaussian Filter', 'Adaptive Threshold', 'Morphological Cleanup',
                            'Contour Detection', 'Shape Classification', 'Object Tracking'],
        'frame_metrics': [
            {
                'frame': m['frame'],
                'time_ms': m['processing_time_ms'],
                'detections': m['n_detections'],
                'contours': m['n_contours'],
            }
            for m in all_metrics
        ],
        'tracking_summary': {
            str(k): {'type': v['type'], 'frames_seen': v['frames_seen']}
            for k, v in pipeline.tracking_history.items()
        },
    }

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: results.json")

    print("\n[5/5] Results Summary:")
    print(f"   Total frames processed:     {total_frames}")
    print(f"   Objects tracked:            {total_objects}")
    print(f"   Avg processing time:        {avg_time:.3f} ms/frame")
    print(f"   Avg detections per frame:   {avg_detections:.1f}")
    print(f"   Pipeline stages:            {len(summary['pipeline_stages'])}")

    print("\n[DONE] Task 4 completed successfully!\n")
    return summary


if __name__ == '__main__':
    main()

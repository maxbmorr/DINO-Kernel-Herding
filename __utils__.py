from pathlib import Path
import os
from PIL import Image
from rembg import remove 
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))
os.environ.setdefault("U2NET_HOME", str(PROJECT_ROOT / ".u2net"))
def rmv_bckgnd(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for image_path in input_root.rglob("*"):
        if image_path.suffix.lower() not in image_extensions:
            continue
        
        relative_path = image_path.relative_to(input_root)

        output_path = output_root / relative_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            img = Image.open(image_path).convert("RGBA")
            no_bg = remove(img)
            no_bg.save(output_path)
            
            print("Saved:", output_path)

        except Exception as e:
            print("Failed", image_path)
            print(e)
    return output_root
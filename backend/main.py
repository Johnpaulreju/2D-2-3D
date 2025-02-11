from fasthtml.common import *
import base64
import io
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision.transforms import functional as TF





model_type = "DPT_Large"  # or "DPT_Hybrid" / "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Get the appropriate transforms for the model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

app, rt = fast_app()

@rt('/')
def home():
    return HTMLResponse("""
    <h1>Upload a 2D Image</h1>
    <form action="/convert" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Convert</button>
    </form>
    """)

@rt('/convert', methods=['POST'])
async def convert(request):
    form = await request.form()
    uploaded_file = form.get('file')
    if not uploaded_file:
        return HTMLResponse("<p>No file uploaded!</p>")
    
    file_bytes = await uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    # Option 1: Create np_image from the PIL image
    np_image = np.array(image)  # Now np_image is defined
    
    image_np = np.array(image)  # Convert the PIL image to a NumPy array
    input_batch = transform(image_np).to(device)

    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=np_image.shape[:2],  # using the np_image dimensions (height, width)
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    # Normalize depth map to 0-255 and convert to uint8
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_uint8 = depth_map_norm.astype(np.uint8)
    
    # Encode original image to base64 for display
    buffered_img = io.BytesIO()
    image.save(buffered_img, format="PNG")
    img_b64 = base64.b64encode(buffered_img.getvalue()).decode("utf-8")
    
    # Encode depth map as an image (for display purposes)
    depth_img = Image.fromarray(depth_map_uint8)
    buffered_depth = io.BytesIO()
    depth_img.save(buffered_depth, format="PNG")
    depth_b64 = base64.b64encode(buffered_depth.getvalue()).decode("utf-8")
    
    # Build HTML that shows the original image on the left and the "depth" image on the right
    html_content = f"""
    <div style="display: flex; justify-content: space-between;">
      <div style="flex: 1; padding: 20px;">
         <h2>Original Image</h2>
         <img src="data:image/png;base64,{img_b64}" style="max-width:100%;">
      </div>
      <div style="flex: 1; padding: 20px;">
         <h2>Estimated Depth Map (Pseudo-3D)</h2>
         <img src="data:image/png;base64,{depth_b64}" style="max-width:100%;">
      </div>
    </div>
    """
    return HTMLResponse(html_content)

serve()



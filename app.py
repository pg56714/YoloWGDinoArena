from typing import List
import cv2
import gradio as gr
import numpy as np
import supervision as sv
from inference.models import YOLOWorld
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
from groundingdino.util.inference import annotate as gd_annotate
from groundingdino.util.inference import predict, load_model
import groundingdino.datasets.transforms as T

MARKDOWN = """
# YoloWGDinoArena

Powered by Roboflow [Inference](https://github.com/roboflow/inference) and 
[Supervision](https://github.com/roboflow/supervision) and [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO).
"""

# GroundingDINO
config_file = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_filenmae = "./weights/groundingdino_swint_ogc.pth"


def image_transform_grounding(init_image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(init_image, None)
    return init_image, image


def image_transform_grounding_for_vis(init_image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
        ]
    )
    image, _ = transform(init_image, None)
    return image


model = load_model(config_file, ckpt_filenmae)


def run_grounding(input_image, grounding_caption, box_threshold, text_threshold):
    init_image = Image.fromarray(input_image.astype("uint8"), "RGB")

    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)

    boxes, logits, phrases = predict(
        model,
        image_tensor,
        grounding_caption,
        box_threshold,
        text_threshold,
        device="cpu",
    )
    annotated_frame = gd_annotate(
        image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases
    )
    image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    return image_with_box


box_threshold = gr.Slider(
    label="Box Threshold",
    minimum=0.0,
    maximum=1.0,
    value=0.25,
    step=0.001,
)
text_threshold = gr.Slider(
    label="Text Threshold",
    minimum=0.0,
    maximum=1.0,
    value=0.25,
    step=0.001,
)

# -----------------------------------------------------------------------------------------------------------

# YOLO-WORLD
# -----------------------------------------------------------------------------------------------------------
YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(",")]


def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = True,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(input_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


def process_image(
    input_image: np.ndarray,
    categories: str,
    confidence_threshold: float,
    nms_threshold: float,
    with_confidence: bool = True,
) -> np.ndarray:
    categories = process_categories(categories)
    YOLO_WORLD_MODEL.set_classes(categories)
    results = YOLO_WORLD_MODEL.infer(input_image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )

    output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_image = annotate_image(
        input_image=output_image,
        detections=detections,
        categories=categories,
        with_confidence=with_confidence,
    )

    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.005,
    step=0.01,
    label="Confidence Threshold",
    # info=(
    #     "The confidence threshold for the YOLO-World model. Lower the threshold to "
    #     "reduce false negatives, enhancing the model's sensitivity to detect "
    #     "sought-after objects. Conversely, increase the threshold to minimize false "
    #     "positives, preventing the model from identifying objects it shouldn't."
    # ),
)

iou_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.1,
    step=0.01,
    label="IoU Threshold",
    # info=(
    #     "The Intersection over Union (IoU) threshold for non-maximum suppression. "
    #     "Decrease the value to lessen the occurrence of overlapping bounding boxes, "
    #     "making the detection process stricter. On the other hand, increase the value "
    #     "to allow more overlapping bounding boxes, accommodating a broader range of "
    #     "detections."
    # ),
)

# -----------------------------------------------------------------------------------------------------------

# View
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        input_image_component = gr.Image(type="numpy", label="Input Image")
        yolo_world_output_image_component = gr.Image(
            type="numpy", label="YOLO-WORLD Output"
        )
        grounding_dion_output_image_component = gr.Image(
            type="pil", label="GroundingDINO Output"
        )
    with gr.Row():
        image_text_component = gr.Textbox(
            label="Categories",
            placeholder="you can input multiple words with comma (,)",
            scale=7,
        )
        submit_button_component = gr.Button(value="Submit", scale=1, variant="primary")

    with gr.Column():
        with gr.Accordion("YOLO-World", open=False):
            confidence_threshold_component.render()
            iou_threshold_component.render()

        with gr.Accordion("GroundingDINO", open=False):
            box_threshold.render()
            text_threshold.render()

    submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            image_text_component,
            confidence_threshold_component,
            iou_threshold_component,
        ],
        outputs=[
            yolo_world_output_image_component,
        ],
    )

    submit_button_component.click(
        fn=run_grounding,
        inputs=[
            input_image_component,
            image_text_component,
            box_threshold,
            text_threshold,
        ],
        outputs=[
            grounding_dion_output_image_component,
        ],
    )

# demo.launch(debug=False, show_error=True, max_threads=1)
demo.launch(debug=False, show_error=True)

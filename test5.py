import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import ollama

# Load YOLO model
model = YOLO("https://app.roboflow.com/ds/elSbLGFAuz?key=m4xWdZn1qet")

# Save global results for question answering
global_analysis = {"boxes": [], "warnings": []}

def calc_distance(box1, box2):
    x1 = (box1[0] + box1[2]) / 2
    y1 = (box1[1] + box1[3]) / 2
    x2 = (box2[0] + box2[2]) / 2
    y2 = (box2[1] + box2[3]) / 2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Analyze image and store results
def analyze_frame(image):
    results = model(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names
    global_analysis["boxes"] = boxes.tolist()
    global_analysis["warnings"] = []

    annotated = image.copy()

    # Draw detections
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = names[classes[i]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{i+1}: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Detect high-risk zones
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            dist = calc_distance(boxes[i], boxes[j])
            if dist < 75:
                global_analysis["warnings"].append(
                    f"Object {i+1} and {j+1} are at risk of collision (distance: {int(dist)} px)")
                cx1 = int((boxes[i][0] + boxes[i][2]) / 2)
                cy1 = int((boxes[i][1] + boxes[i][3]) / 2)
                cx2 = int((boxes[j][0] + boxes[j][2]) / 2)
                cy2 = int((boxes[j][1] + boxes[j][3]) / 2)
                cv2.line(annotated, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)

    summary = "\n".join(global_analysis["warnings"]) if global_analysis["warnings"] else "No collision risks detected."
    return annotated, summary

# Ask LLaMA a question based on the detection data
def ask_question(question):
    prompt = f"""Given the following object detection results:

Boxes: {global_analysis["boxes"]}
Warnings: {global_analysis["warnings"]}

Answer this user question: {question}
"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## ✈️ Drone Object Detection and Reasoning Assistant\n"
        "### Upload a drone image frame to detect objects and predict potential collisions."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="numpy",
                label=" Drone Frame (Drop or click to upload)",
                height=320
            )
            analyze_btn = gr.Button(" Analyze Frame", size="lg")

        with gr.Column(scale=2):
            image_output = gr.Image(label=" Detection Output", height=320)
            warning_output = gr.Textbox(
                label="⚠️ Collision Warnings",
                lines=4,
                interactive=False,
                show_copy_button=True,
                placeholder="Collision warnings will appear here..."
            )

    with gr.Accordion(" Ask the Scene Analyzer", open=False):
        with gr.Row():
            question_input = gr.Textbox(
                label=" Ask a question",
                placeholder="Example: Are any objects too close to each other?"
            )
            ask_btn = gr.Button("Ask LLaMA")
        answer_output = gr.Textbox(
            label="🤖 LLaMA Answer",
            lines=4,
            interactive=False,
            show_copy_button=True,
            placeholder="LLaMA's answer will appear here based on detection results."
        )

    # Connect buttons to functions
    analyze_btn.click(fn=analyze_frame, inputs=image_input, outputs=[image_output, warning_output])
    ask_btn.click(fn=ask_question, inputs=question_input, outputs=answer_output)

demo.launch()
demo.launch(share=True)

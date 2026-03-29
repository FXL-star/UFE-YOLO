from ultralytics import YOLO

# Create a new YOLO model from scratch
# fileName = "best"
fileName = "best"
model = YOLO(f"./images/{fileName}.pt")

results = model("./images/QQ20250804-180243.png")

for result in results:
    result.show()  # display to screen
    result.save(filename=f"{fileName}result.jpg")  # save to disk
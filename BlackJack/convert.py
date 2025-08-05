import glob
import os
import xml.etree.ElementTree as ET

# --- Define your classes here ---
# This must match the order in your data.yaml file
classes = ['a','2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k']


def xml_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    yolo_lines = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)

        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

        # YOLO format: (class_id, x_center, y_center, width, height) - all normalized
        x_center = (b[0] + b[1]) / 2.0 / w
        y_center = (b[2] + b[3]) / 2.0 / h
        width = (b[1] - b[0]) / w
        height = (b[3] - b[2]) / h

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write to .txt file
    base_filename = os.path.splitext(os.path.basename(xml_file))[0]
    output_filepath = os.path.join(output_dir, f"{base_filename}.txt")
    with open(output_filepath, "w") as f:
        f.write("\n".join(yolo_lines))


# --- Main conversion logic ---
# Create output directories if they don't exist
os.makedirs('single_dataset/labels/train', exist_ok=True)
os.makedirs('single_dataset/labels/val', exist_ok=True)

# Convert training annotations
# Replace 'path/to/your/train_xmls' with the path to your temporary folder of train XMLs
train_xml_dir = 'single_dataset/images/TrainXml/*.xml'
for xml_file in glob.glob(train_xml_dir):
    xml_to_yolo(xml_file, 'single_dataset/labels/Train')

# Convert validation annotations
# Replace 'path/to/your/val_xmls' with the path to your temporary folder of validation XMLs
val_xml_dir = 'single_dataset/images/ValXml/*.xml'
for xml_file in glob.glob(val_xml_dir):
    xml_to_yolo(xml_file, 'single_dataset/labels/Val')

print("Conversion complete!")